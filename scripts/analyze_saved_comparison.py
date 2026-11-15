"""Add a train-fitted statistical baseline and uncertainty analysis to saved forecasts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import chronological_split, load_hourly_csv
from src.evaluation_analysis import paired_block_bootstrap, period_metrics
from src.error_analysis import error_slices
from src.metrics import score
from src.provenance import artifact_info, build_run_manifest
from src.run_experiment import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV containing timestamp,load columns")
    parser.add_argument("--predictions", required=True,
                        help="Saved comparison predictions.csv with model,window_start,timestamp,lead_hour,actual,forecast")
    parser.add_argument("--output-dir", help="Defaults to the predictions file's directory")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--train-end", help="Exclusive timestamp boundary for training")
    parser.add_argument("--validation-end", help="Exclusive timestamp boundary for validation")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")

    series = load_hourly_csv(args.data)
    splits = chronological_split(series, train_end=args.train_end,
                                 validation_end=args.validation_end)
    scale = float(np.mean(np.abs(splits.train.to_numpy()[24:] - splits.train.to_numpy()[:-24])))
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent
    predictions = pd.read_csv(predictions_path, parse_dates=["window_start", "timestamp"])
    source_predictions = artifact_info(predictions_path)
    required = {"model", "window_start", "timestamp", "lead_hour", "actual", "forecast"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    if "hour_of_week_profile" not in set(predictions["model"]):
        baseline_metrics, baseline = evaluate(
            series, "hour_of_week_profile", args.horizon, args.context_length,
            max_windows=0, split_name=args.split, train_end=args.train_end,
            validation_end=args.validation_end,
        )
        reference = predictions[predictions["model"] == predictions["model"].iloc[0]]
        baseline_index = pd.MultiIndex.from_frame(baseline[["window_start", "timestamp"]])
        reference_index = pd.MultiIndex.from_frame(reference[["window_start", "timestamp"]])
        if not baseline_index.equals(reference_index):
            raise ValueError(
                "Saved forecasts do not match all daily origins; regenerate them together before adding the baseline"
            )
        if not np.array_equal(baseline["actual"].to_numpy(), reference["actual"].to_numpy()):
            raise ValueError("Saved forecasts and baseline have different actual values")
        baseline.insert(0, "model", "hour_of_week_profile")
        predictions = pd.concat([predictions, baseline], ignore_index=True)

    order = ["seasonal_naive", "hour_of_week_profile", "timesfm_zero_shot", "timesfm_lora"]
    models = [name for name in order if name in set(predictions["model"])]
    if len(models) < 3:
        raise ValueError(f"Expected the shared comparison models; found: {models}")
    reference_frame = predictions[predictions["model"] == models[0]]
    ref_index = pd.MultiIndex.from_frame(reference_frame[["window_start", "timestamp"]])
    for model_name in models[1:]:
        group = predictions[predictions["model"] == model_name]
        idx = pd.MultiIndex.from_frame(group[["window_start", "timestamp"]])
        if not idx.equals(ref_index) or not np.array_equal(
            group["actual"].to_numpy(), reference_frame["actual"].to_numpy()
        ):
            raise ValueError("Comparison models must share identical forecast origins and actual values")

    summary = []
    for model_name in models:
        group = predictions[predictions["model"] == model_name]
        metrics = score(group["actual"], group["forecast"], scale)
        summary.append({
            "model": model_name,
            "evaluation_split": args.split,
            "metrics": metrics,
            "horizon_hours": args.horizon,
            "context_length": args.context_length,
            "windows_evaluated": int(group["window_start"].nunique()),
            "hours_evaluated": int(len(group)),
            "seasonal_scale": scale,
        })
    mae = {result["model"]: result["metrics"]["MAE"] for result in summary}
    for result in summary:
        result["MAE_skill_vs_seasonal_naive"] = 1.0 - result["metrics"]["MAE"] / mae["seasonal_naive"]
        result["MAE_skill_vs_hour_of_week_profile"] = (
            1.0 - result["metrics"]["MAE"] / mae["hour_of_week_profile"]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = predictions.sort_values(["model", "window_start", "lead_hour"])
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([
        {"model": result["model"], **result["metrics"],
         "MAE_skill_vs_seasonal_naive": result["MAE_skill_vs_seasonal_naive"],
         "MAE_skill_vs_hour_of_week_profile": result["MAE_skill_vs_hour_of_week_profile"],
         "windows_evaluated": result["windows_evaluated"],
         "hours_evaluated": result["hours_evaluated"]}
        for result in summary
    ]).to_csv(output_dir / "summary.csv", index=False)

    lead_rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        errors = group["forecast"].to_numpy() - group["actual"].to_numpy()
        by_lead = group.assign(abs_error=np.abs(errors), squared_error=errors ** 2)
        for lead_hour, lead in by_lead.groupby("lead_hour", sort=True):
            lead_rows.append({
                "model": model_name,
                "lead_hour": int(lead_hour),
                "MAE": float(lead["abs_error"].mean()),
                "RMSE": float(np.sqrt(lead["squared_error"].mean())),
                "hours": int(len(lead)),
            })
    pd.DataFrame(lead_rows).to_csv(output_dir / "lead_metrics.csv", index=False)
    period_metrics(predictions, scale).to_csv(output_dir / "period_metrics.csv", index=False)
    error_slices(predictions, splits.train).to_csv(output_dir / "error_slices.csv", index=False)

    uncertainty = {
        "method": "paired moving-block bootstrap over daily forecast-origin MAE",
        "block_days": 7,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": 42,
        "interpretation": "Negative MAE difference and positive MAE reduction favor the candidate model.",
        "comparisons": {},
    }
    pairs = [("timesfm_lora", "timesfm_zero_shot"),
             ("timesfm_lora", "hour_of_week_profile"),
             ("timesfm_lora", "seasonal_naive")]
    for candidate, reference in pairs:
        try:
            uncertainty["comparisons"][candidate + "_vs_" + reference] = paired_block_bootstrap(
                predictions, candidate, reference, args.horizon,
                replicates=args.bootstrap_replicates, seed=42,
            )
        except ValueError as exc:
            uncertainty["comparisons"][candidate + "_vs_" + reference] = {
                "not_computed_reason": str(exc)
            }
    (output_dir / "uncertainty.json").write_text(json.dumps(uncertainty, indent=2) + "\n")

    analysis_manifest = build_run_manifest(
        data_path=args.data,
        series=series,
        splits=splits,
        args=vars(args),
        model_id="google/timesfm-2.5-200m-transformers",
        adapter_path=None,
        operation="analysis_of_saved_forecasts",
        extra={
            "inference_performed": False,
            "source_predictions": source_predictions,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": 42,
        },
    )
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2) + "\n"
    )

    first_origin = predictions["window_start"].min()
    first_window = predictions[predictions["window_start"] == first_origin]
    fig, ax = plt.subplots(figsize=(12, 4))
    actual = first_window[first_window["model"] == models[0]]
    ax.plot(actual["timestamp"], actual["actual"], label="Actual", linewidth=2)
    for model_name in models:
        group = first_window[first_window["model"] == model_name]
        ax.plot(group["timestamp"], group["forecast"], label=model_name, linewidth=1.3)
    ax.set(title=f"{args.split.title()} forecast comparison ({args.horizon} hours)",
           ylabel="Load", xlabel="Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"models": summary, "uncertainty": uncertainty}, indent=2))


if __name__ == "__main__":
    main()
