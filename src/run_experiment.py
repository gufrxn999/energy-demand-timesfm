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
             max_windows: int = 0, adapter: str | None = None,
             split_name: str = "validation", inference_batch_size: int = 8,
             model_id: str = "google/timesfm-2.5-200m-transformers"
             ) -> tuple[dict, pd.DataFrame]:
    splits = chronological_split(series)
    if horizon < 1 or context_length < 1:
        raise ValueError("horizon and context length must be positive")
    if max_windows < 0:
        raise ValueError("windows must be zero (all daily windows) or a positive number")
    if split_name not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    part = splits.validation if split_name == "validation" else splits.test
    model = (TimesFMZeroShot(context_length, horizon, adapter, model_id)
             if model_name == "timesfm" else None)

    if len(part) < horizon:
        raise ValueError(f"{split_name} segment is shorter than the forecast horizon")
    # Daily, non-overlapping forecasts cover the selected period. A positive
    # max_windows samples evenly from those daily origins; zero evaluates all.
    starts = np.arange(0, len(part) - horizon + 1, horizon, dtype=int)
    if max_windows and len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts) - 1, max_windows, dtype=int)]
    contexts, actual_all, records = [], [], []
    for start in starts:
        absolute = series.index.get_loc(part.index[start])
        context_start = max(0, absolute - context_length)
        context = series.iloc[context_start:absolute].to_numpy(dtype=np.float32)
        if len(context) < 24:
            continue
        contexts.append(context)
        actual = part.iloc[start:start + horizon].to_numpy(dtype=float)
        actual_all.extend(actual)
        records.extend(
            (part.index[start], ts, lead_hour, float(a))
            for lead_hour, (ts, a) in enumerate(
                zip(part.index[start:start + horizon], actual)
            )
        )
    if not contexts:
        raise ValueError("No forecast windows have enough history")
    if model:
        predicted = model.predict_many(contexts, inference_batch_size)
    else:
        predicted = np.stack([seasonal_naive(context, horizon) for context in contexts])
    predicted_all = predicted.reshape(-1)
    predictions = pd.DataFrame(
        records, columns=["window_start", "timestamp", "lead_hour", "actual"]
    )
    predictions["forecast"] = predicted_all
    scale = float(np.mean(np.abs(splits.train.to_numpy()[24:] - splits.train.to_numpy()[:-24])))
    metrics = {
        "model": model_name,
        "evaluation_split": split_name,
        "metrics": score(actual_all, predicted_all, scale),
        "horizon_hours": horizon,
        "context_length": context_length,
        "windows_evaluated": len(contexts),
        "hours_evaluated": len(predicted_all),
    }
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV containing timestamp,load columns")
    parser.add_argument("--model", choices=["seasonal_naive", "timesfm"], default="seasonal_naive")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--windows", type=int, default=0,
                        help="Daily forecast origins (0 evaluates every daily origin)")
    parser.add_argument("--split", choices=["validation", "test"], default="validation",
                        help="Keep test for a final comparison after model selection")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--model-id", default="google/timesfm-2.5-200m-transformers",
                        help="Local checkpoint path or Hugging Face model ID")
    parser.add_argument("--adapter", help="Optional TimesFM 2.5 PEFT adapter directory")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    series = load_hourly_csv(args.data)
    metrics, predictions = evaluate(series, args.model, args.horizon, args.context_length,
                                   args.windows, args.adapter, args.split,
                                   args.inference_batch_size, args.model_id)
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
