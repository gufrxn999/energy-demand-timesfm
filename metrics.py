"""Forecast metrics with safe handling of zero denominators."""
import numpy as np


def score(actual, predicted, seasonal_scale: float) -> dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mase = mae / seasonal_scale if seasonal_scale > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MASE": float(mase)}
