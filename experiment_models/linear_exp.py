import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import preprocess_data
from feature_engineering import build_features
from models.regression.linear import (
    train_linear_regressor,
    evaluate_linear_regressor
)

df = preprocess_data("problems_dataset.jsonl")

X_train, X_test, idx_train, idx_test, tfidf, scaler = build_features(df)

y_train_score = df.loc[idx_train, "problem_score"]
y_test_score = df.loc[idx_test, "problem_score"]

regressor = train_linear_regressor(X_train, y_train_score)

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