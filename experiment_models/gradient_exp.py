from data_preprocessing import preprocess_data
from feature_engineering import build_features
from models.regression.gradient_boosting import (
    train_gb_regressor,
    evaluate_gb_regressor
)

df = preprocess_data("problems_dataset.jsonl")

X_train, X_test, idx_train, idx_test, tfidf, scaler = build_features(df)

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