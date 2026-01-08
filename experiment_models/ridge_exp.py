import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import preprocess_data
from feature_engineering import build_features
from models.classification.logistic import (
    train_logistic_classifier,
    evaluate_logistic_classifier
)
from models.regression.ridge import train_ridge_regressor, evaluate_ridge_regressor

df = preprocess_data("problems_dataset.jsonl")

X_train, X_test, idx_train, idx_test, tfidf, scaler = build_features(df)

y_train_score = df.loc[idx_train, "problem_score"]
y_test_score = df.loc[idx_test, "problem_score"]

for alpha in [0.1, 1, 5, 10, 50]:
    ridge = train_ridge_regressor(X_train, y_train_score, alpha=alpha)

    ridge_results = evaluate_ridge_regressor(
        ridge,
        X_test,
        y_test_score,
        y_train_mean=y_train_score.mean()
    )

    print(
        f"alpha={alpha} | "
        f"MAE={ridge_results['mae']:.2f} | "
        f"RMSE={ridge_results['rmse']:.2f}"
    )