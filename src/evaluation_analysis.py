"""Time-period summaries and paired uncertainty estimates for forecast tables."""
import math

import numpy as np
import pandas as pd


def period_metrics(predictions: pd.DataFrame, seasonal_scale: float,
                   frequency: str = "Q") -> pd.DataFrame:
    """Summarize forecast errors by calendar quarter (or another pandas period)."""
    required = {"model", "window_start", "timestamp", "actual", "forecast"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["period"] = frame["timestamp"].dt.to_period(frequency).astype(str)
    rows = []
    for (model, period), group in frame.groupby(["model", "period"], sort=True):
        errors = group["forecast"].to_numpy(dtype=float) - group["actual"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(errors)))
        rows.append({
            "model": model,
            "period": period,
            "period_start": group["timestamp"].min(),
            "period_end": group["timestamp"].max(),
            "windows": int(group["window_start"].nunique()),
            "hours": int(len(group)),
            "MAE": mae,
            "RMSE": float(np.sqrt(np.mean(errors ** 2))),
            "MASE": float(mae / seasonal_scale) if seasonal_scale > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def paired_block_bootstrap(predictions: pd.DataFrame, candidate: str,
                           reference: str, horizon_hours: int = 24,
                           block_days: int = 7, replicates: int = 2000,
                           seed: int = 42) -> dict[str, float | int | str]:
    """Bootstrap paired daily MAEs using contiguous moving blocks of days.

    Forecast origins must be the full, consecutive daily sequence and all
    compared windows must contain the same number of forecast hours.
    """
    if block_days < 1 or replicates < 1 or horizon_hours < 1:
        raise ValueError("block_days, replicates, and horizon_hours must be positive")
    subset = predictions[predictions["model"].isin([candidate, reference])].copy()
    if set(subset["model"].unique()) != {candidate, reference}:
        raise ValueError("Both candidate and reference models must be present")
    subset["window_start"] = pd.to_datetime(subset["window_start"])
    subset["abs_error"] = np.abs(
        subset["forecast"].to_numpy(dtype=float) - subset["actual"].to_numpy(dtype=float)
    )
    daily = subset.groupby(["window_start", "model"])["abs_error"].mean().unstack("model")
    counts = subset.groupby(["window_start", "model"]).size().unstack("model")
    if daily[[candidate, reference]].isna().any().any():
        raise ValueError("Models must have a paired forecast for every origin")
    if not (counts[candidate] == counts[reference]).all():
        raise ValueError("Models must have the same number of forecast hours at each origin")
    origins = pd.DatetimeIndex(daily.index).sort_values()

    candidate_mae = daily.loc[origins, candidate].to_numpy(dtype=float)
    reference_mae = daily.loc[origins, reference].to_numpy(dtype=float)
    origin_gaps = pd.Series(origins).diff().iloc[1:]
    expected_gap = pd.Timedelta(hours=horizon_hours)
    if not origin_gaps.eq(expected_gap).all():
        raise ValueError("Block bootstrap requires complete consecutive forecast origins; use --windows 0")
    count = len(origins)
    if count < 2:
        raise ValueError("At least two paired forecast origins are required")
    block = min(block_days, count)
    block_starts = np.arange(count - block + 1)
    blocks_per_sample = math.ceil(count / block)
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)
    reductions = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        starts = rng.choice(block_starts, size=blocks_per_sample, replace=True)
        indices = (starts[:, None] + np.arange(block)).reshape(-1)[:count]
        candidate_score = float(np.mean(candidate_mae[indices]))
        reference_score = float(np.mean(reference_mae[indices]))
        differences[replicate] = candidate_score - reference_score
        reductions[replicate] = (
            100.0 * (1.0 - candidate_score / reference_score)
            if reference_score > 0 else float("nan")
        )

    point_candidate = float(np.mean(candidate_mae))
    point_reference = float(np.mean(reference_mae))
    return {
        "candidate": candidate,
        "reference": reference,
        "origins": int(count),
        "block_days": int(block),
        "candidate_mae": point_candidate,
        "reference_mae": point_reference,
        "candidate_minus_reference_mae": point_candidate - point_reference,
        "mae_difference_ci_95_lower": float(np.quantile(differences, 0.025)),
        "mae_difference_ci_95_upper": float(np.quantile(differences, 0.975)),
        "mae_reduction_pct": (
            100.0 * (1.0 - point_candidate / point_reference)
            if point_reference > 0 else float("nan")
        ),
        "mae_reduction_ci_95_lower_pct": float(np.quantile(reductions, 0.025)),
        "mae_reduction_ci_95_upper_pct": float(np.quantile(reductions, 0.975)),
        "daily_win_rate": float(np.mean(candidate_mae < reference_mae)),
        "bootstrap_replicates": int(replicates),
        "seed": int(seed),
    }
