"""Input loading, hourly regularization, and leakage-safe time splits."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataSplits:
    train: pd.Series
    validation: pd.Series
    test: pd.Series


def load_hourly_csv(path: str | Path) -> pd.Series:
    frame = pd.read_csv(path)
    missing = {"timestamp", "load"} - set(frame.columns)
    if missing:
        raise ValueError(f"CSV must contain timestamp and load columns; missing: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["load"] = pd.to_numeric(frame["load"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "load"]).sort_values("timestamp")
    if frame.empty:
        raise ValueError("No valid timestamp/load rows found")
    series = frame.drop_duplicates("timestamp", keep="last").set_index("timestamp")["load"].astype(float)
    series = series.resample("h").mean().interpolate(method="time", limit=3, limit_direction="both")
    series = series.dropna()
    if len(series) < 24 * 15:
        raise ValueError("Need at least 15 days of hourly data for this experiment")
    series.name = "load"
    return series


def chronological_split(series: pd.Series, train_fraction: float = 0.7,
                        validation_fraction: float = 0.15) -> DataSplits:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation fractions must be below 1")
    train_end = int(len(series) * train_fraction)
    validation_end = int(len(series) * (train_fraction + validation_fraction))
    return DataSplits(series.iloc[:train_end], series.iloc[train_end:validation_end], series.iloc[validation_end:])
