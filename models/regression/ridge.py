from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def train_ridge_regressor(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def evaluate_ridge_regressor(model, X_test, y_test, y_train_mean=None):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results = {
        "mae": mae,
        "rmse": rmse
    }

    if y_train_mean is not None:
        baseline_pred = np.full(len(y_test), y_train_mean)
        results["baseline_mae"] = mean_absolute_error(y_test, baseline_pred)
        results["baseline_rmse"] = np.sqrt(mean_squared_error(y_test, baseline_pred))

    return results