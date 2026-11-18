"""Training and evaluation for the Parcel Tracker Estimator."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "delivery_data.csv"
MODEL_DIR = ROOT / "models"

FEATURES = ["distance_km", "package_weight_kg", "courier_type"]
CONTINUOUS_TARGET = "continuous_delivery_days"
DISCRETE_TARGET = "discrete_delivery_days"


def _pipeline(estimator):
    preprocessing = ColumnTransformer(
        transformers=[
            ("courier", OneHotEncoder(handle_unknown="ignore"), ["courier_type"]),
            ("numeric", StandardScaler(), ["distance_km", "package_weight_kg"]),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])


def _metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def train_models():
    """Train both models on one shared, held-out 80/20 split."""
    frame = pd.read_csv(DATA_PATH)
    required = FEATURES + [CONTINUOUS_TARGET, DISCRETE_TARGET]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    if frame[required].isna().any().any():
        raise ValueError("Training data contains missing model values.")
    if (frame["distance_km"] <= 0).any() or (frame["package_weight_kg"] <= 0).any():
        raise ValueError("Distance and package weight must be positive.")
    if (frame[CONTINUOUS_TARGET] <= 0).any():
        raise ValueError("Continuous delivery durations must be positive.")
    if (frame[DISCRETE_TARGET] < 1).any():
        raise ValueError("Discrete delivery-day counts must be positive integers.")
    if not np.allclose(frame[DISCRETE_TARGET], np.round(frame[DISCRETE_TARGET])):
        raise ValueError("Discrete target must contain whole-day counts.")

    X = frame[FEATURES].copy()
    y_cont = frame[CONTINUOUS_TARGET].astype(float)
    y_disc = frame[DISCRETE_TARGET].astype(int)

    X_train, X_test, yc_train, yc_test, yd_train, yd_test = train_test_split(
        X, y_cont, y_disc, test_size=0.20, random_state=42
    )

    linear_model = _pipeline(LinearRegression())
    poisson_model = _pipeline(PoissonRegressor(alpha=0.08, max_iter=3000))

    linear_model.fit(X_train, yc_train)
    poisson_model.fit(X_train, yd_train)

    linear_pred = np.maximum(0.1, linear_model.predict(X_test))
    poisson_pred = np.maximum(0.1, poisson_model.predict(X_test))

    # Mean-only baselines make it possible to see whether the models add value.
    linear_baseline = np.full(len(yc_test), float(yc_train.mean()))
    poisson_baseline = np.full(len(yd_test), float(yd_train.mean()))

    metrics = {
        "linear": _metrics(yc_test, linear_pred),
        "poisson": _metrics(yd_test, poisson_pred),
        "linear_baseline_mae": float(mean_absolute_error(yc_test, linear_baseline)),
        "poisson_baseline_mae": float(mean_absolute_error(yd_test, poisson_baseline)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(frame)),
    }

    # Save compatible models locally for inspection/reuse; the app trains in its
    # current environment and does not depend on old serialized files.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(linear_model, MODEL_DIR / "linear_model.joblib")
    joblib.dump(poisson_model, MODEL_DIR / "poisson_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    test_predictions = pd.DataFrame({
        "actual_continuous": yc_test.to_numpy(),
        "predicted_continuous": linear_pred,
        "actual_discrete": yd_test.to_numpy(),
        "predicted_discrete": poisson_pred,
    }).reset_index(drop=True)

    return linear_model, poisson_model, metrics, test_predictions


if __name__ == "__main__":
    _, _, result, _ = train_models()
    print("Training completed successfully.")
    print(f"Training records: {result['train_rows']:,}")
    print(f"Test records: {result['test_rows']:,}")
    print("\\nMultiple Linear Regression")
    for key, value in result["linear"].items():
        print(f"{key}: {value:.3f}")
    print("\\nPoisson Regression")
    for key, value in result["poisson"].items():
        print(f"{key}: {value:.3f}")
