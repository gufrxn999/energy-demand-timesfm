"""Leakage-aware operational slices for saved forecast errors."""
import numpy as np
import pandas as pd


def error_slices(predictions: pd.DataFrame, training_values,
                 high_load_quantile: float = 0.9) -> pd.DataFrame:
    """Report error by hour, weekday/weekend, and training-defined load regime."""
    required = {"model", "timestamp", "actual", "forecast"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if not 0 < high_load_quantile < 1:
        raise ValueError("high_load_quantile must be between zero and one")
    train = np.asarray(training_values, dtype=float)
    train = train[np.isfinite(train)]
    if not len(train):
        raise ValueError("Training values must contain at least one finite value")

    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["error"] = frame["forecast"].astype(float) - frame["actual"].astype(float)
    frame["abs_error"] = frame["error"].abs()
    frame["hour_of_day"] = frame["timestamp"].dt.hour.astype(str).str.zfill(2)
    frame["day_type"] = np.where(frame["timestamp"].dt.dayofweek < 5, "weekday", "weekend")
    threshold = float(np.quantile(train, high_load_quantile))
    frame["load_regime"] = np.where(
        frame["actual"] > threshold,
        f"above_train_p{int(high_load_quantile * 100)}",
        f"at_or_below_train_p{int(high_load_quantile * 100)}",
    )

    rows = []
    dimensions = (
        ("hour_of_day", "hour_of_day"),
        ("day_type", "day_type"),
        ("load_regime", "load_regime"),
    )
    for dimension, column in dimensions:
        for (model, segment), group in frame.groupby(["model", column], sort=True, observed=True):
            errors = group["error"].to_numpy(dtype=float)
            rows.append({
                "dimension": dimension,
                "segment": str(segment),
                "model": model,
                "hours": int(len(group)),
                "share_of_model_hours": float(len(group) / frame[frame["model"] == model].shape[0]),
                "MAE": float(np.mean(np.abs(errors))),
                "RMSE": float(np.sqrt(np.mean(errors ** 2))),
                "mean_error": float(np.mean(errors)),
                "p90_absolute_error": float(np.quantile(np.abs(errors), 0.9)),
                "training_load_quantile": float(high_load_quantile),
                "training_load_threshold": threshold,
            })
    result = pd.DataFrame(rows)
    result.attrs["high_load_threshold"] = threshold
    result.attrs["high_load_quantile"] = high_load_quantile
    return result
