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
    # Carry observations forward only: interpolation can use future values and
    # leak across a chronological train/validation/test boundary.
    series = series.resample("h").mean().ffill(limit=3)
    if series.isna().any():
        count = int(series.isna().sum())
        first_gap = series.index[series.isna()][0]
        raise ValueError(
            f"Found {count} missing hourly values after forward-filling gaps of up to 3 hours; "
            f"first unresolved gap is at {first_gap}"
        )
    if len(series) < 24 * 15:
        raise ValueError("Need at least 15 days of hourly data for this experiment")
    series.name = "load"
    return series


def chronological_split(series: pd.Series, train_fraction: float = 0.7,
                        validation_fraction: float = 0.15,
                        train_end: str | pd.Timestamp | None = None,
                        validation_end: str | pd.Timestamp | None = None,
                        test_end: str | pd.Timestamp | None = None) -> DataSplits:
    if train_end is not None or validation_end is not None or test_end is not None:
        if train_end is None or validation_end is None:
            raise ValueError("Both train_end and validation_end are required for date-based splits")
        train_boundary = pd.Timestamp(train_end)
        validation_boundary = pd.Timestamp(validation_end)
        test_boundary = pd.Timestamp(test_end) if test_end is not None else None
        if series.index.tz is not None:
            def align_timezone(boundary):
                if boundary is None:
                    return None
                return (boundary.tz_localize(series.index.tz) if boundary.tzinfo is None
                        else boundary.tz_convert(series.index.tz))
            train_boundary = align_timezone(train_boundary)
            validation_boundary = align_timezone(validation_boundary)
            test_boundary = align_timezone(test_boundary)
        elif any(boundary is not None and boundary.tzinfo is not None
                 for boundary in (train_boundary, validation_boundary, test_boundary)):
            raise ValueError(
                "Timezone-aware date boundaries require a timezone-aware series index"
            )
        if train_boundary >= validation_boundary:
            raise ValueError("train_end must be earlier than validation_end")
        if test_boundary is not None and test_boundary <= validation_boundary:
            raise ValueError("test_end must be later than validation_end")
        splits = DataSplits(
            series.loc[series.index < train_boundary],
            series.loc[(series.index >= train_boundary) & (series.index < validation_boundary)],
            series.loc[(series.index >= validation_boundary) &
                       ((series.index < test_boundary) if test_boundary is not None else True)],
        )
        if any(part.empty for part in (splits.train, splits.validation, splits.test)):
            raise ValueError("Date boundaries must leave at least one observation in each split")
        return splits
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation fractions must be below 1")
    train_end = int(len(series) * train_fraction)
    validation_end = int(len(series) * (train_fraction + validation_fraction))
    return DataSplits(series.iloc[:train_end], series.iloc[train_end:validation_end], series.iloc[validation_end:])
