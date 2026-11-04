"""Rolling-origin evaluation for hourly load forecasts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import chronological_split, load_hourly_csv
from .metrics import score
from .models import TimesFMZeroShot, seasonal_naive


def evaluate(series: pd.Series, model_name: str, horizon: int, context_length: int,
             max_windows: int, adapter: str | None = None) -> tuple[dict, pd.DataFrame]:
    splits = chronological_split(series)
    # Validation selects the run configuration; test period is reserved for the final report.
    validation = splits.validation
    test = splits.test
    stride = horizon
    model = TimesFMZeroShot(context_length, horizon, adapter) if model_name == "timesfm" else None

    def run(part: pd.Series):
        if len(part) <= horizon:
            raise ValueError("Validation/test segment is shorter than forecast horizon")
        n = min(max_windows, max(1, (len(part) - horizon) // stride))
        starts = np.linspace(0, len(part) - horizon, n, dtype=int)
        records, actual_all, predicted_all = [], [], []
        for start in starts:
            absolute = series.index.get_loc(part.index[start])
            context_start = max(0, absolute - context_length)
            context = series.iloc[context_start:absolute]
            if len(context) < 24:
                continue
            pred = (model.predict(context) if model else seasonal_naive(context, horizon))
            actual = part.iloc[start:start + horizon].to_numpy()
            records.extend({"timestamp": ts, "actual": float(a), "forecast": float(p)}
                           for ts, a, p in zip(part.index[start:start + horizon], actual, pred))
            actual_all.extend(actual)
            predicted_all.extend(pred)
        scale = float(np.mean(np.abs(splits.train.to_numpy()[24:] - splits.train.to_numpy()[:-24])))
        return score(actual_all, predicted_all, scale), pd.DataFrame(records)

    validation_metrics, _ = run(validation)
    test_metrics, predictions = run(test)
    return {"model": model_name, "validation": validation_metrics, "test": test_metrics,
            "horizon_hours": horizon, "context_length": context_length,
            "validation_windows": max_windows}, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV containing timestamp,load columns")
    parser.add_argument("--model", choices=["seasonal_naive", "timesfm"], default="seasonal_naive")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--windows", type=int, default=14)
    parser.add_argument("--adapter", help="Optional TimesFM 2.5 PEFT adapter directory")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    series = load_hourly_csv(args.data)
    metrics, predictions = evaluate(series, args.model, args.horizon, args.context_length,
                                   args.windows, args.adapter)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions.to_csv(out / "predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    # Evaluation windows can be spread across the test period. Show a single
    # contiguous forecast window rather than connecting separate dates in a line.
    plot = predictions.iloc[:min(len(predictions), args.horizon)]
    ax.plot(plot["timestamp"], plot["actual"], label="Actual", linewidth=1.8)
    ax.plot(plot["timestamp"], plot["forecast"], label=args.model, linewidth=1.5)
    ax.set(title=f"Hourly demand: {args.horizon}-hour held-out test forecast",
           ylabel="Load", xlabel="Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "forecast.png", dpi=160)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
