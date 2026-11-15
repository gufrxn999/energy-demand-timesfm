"""Compare statistical baselines, zero-shot TimesFM, and a LoRA adapter."""
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
from src.evaluation_analysis import period_metrics, paired_block_bootstrap
from src.error_analysis import error_slices
from src.provenance import build_run_manifest
from src.run_experiment import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with timestamp,load columns")
    parser.add_argument("--adapter", required=True, help="Fine-tuned PEFT adapter directory")
    parser.add_argument("--model-id", default="google/timesfm-2.5-200m-transformers",
                        help="Local checkpoint path or Hugging Face model ID")
    parser.add_argument("--split", choices=["validation", "test"], default="validation",
                        help="Use test only for a final comparison after model selection")
    parser.add_argument("--train-end", help="Exclusive timestamp boundary for training")
    parser.add_argument("--validation-end", help="Exclusive timestamp boundary for validation")
    parser.add_argument("--test-end", help="Optional exclusive timestamp boundary for test")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--windows", type=int, default=0,
                        help="Daily forecast origins (0 evaluates every daily origin)")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/model-comparison")
    args = parser.parse_args()

    series = load_hourly_csv(args.data)
    comparisons = [
        ("seasonal_naive", "seasonal_naive", None),
        ("hour_of_week_profile", "hour_of_week_profile", None),
        ("timesfm_zero_shot", "timesfm", None),
        ("timesfm_lora", "timesfm", args.adapter),
    ]
    summary, forecasts = [], []
    reference = None
    for label, model_name, adapter in comparisons:
        result, prediction = evaluate(
            series, model_name, args.horizon, args.context_length,
            args.windows, adapter, args.split, args.inference_batch_size,
            args.model_id, args.train_end, args.validation_end,
            args.test_end,
        )
        actual_index = pd.DatetimeIndex(prediction["timestamp"])
        actual_values = prediction["actual"].to_numpy()
        if reference is None:
            reference = (actual_index, actual_values)
        elif not actual_index.equals(reference[0]) or not np.array_equal(actual_values, reference[1]):
            raise RuntimeError("Models were not scored on the same timestamps and actual values")
        result["model"] = label
        summary.append(result)
        prediction.insert(0, "model", label)
        forecasts.append(prediction)
        print(json.dumps(result, indent=2), flush=True)

    mae_by_model = {result["model"]: result["metrics"]["MAE"] for result in summary}
    seasonal_mae = mae_by_model["seasonal_naive"]
    profile_mae = mae_by_model["hour_of_week_profile"]
    for result in summary:
        result["MAE_skill_vs_seasonal_naive"] = (
            1.0 - result["metrics"]["MAE"] / seasonal_mae if seasonal_mae > 0 else float("nan")
        )
        result["MAE_skill_vs_hour_of_week_profile"] = (
            1.0 - result["metrics"]["MAE"] / profile_mae if profile_mae > 0 else float("nan")
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = chronological_split(series, train_end=args.train_end,
                                 validation_end=args.validation_end,
                                 test_end=args.test_end)
    manifest = build_run_manifest(
        data_path=args.data,
        series=series,
        splits=splits,
        args=vars(args),
        model_id=args.model_id,
        adapter_path=args.adapter,
    )
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([
        {"model": result["model"], **result["metrics"],
         "MAE_skill_vs_seasonal_naive": result["MAE_skill_vs_seasonal_naive"],
         "MAE_skill_vs_hour_of_week_profile": result["MAE_skill_vs_hour_of_week_profile"],
         "windows_evaluated": result["windows_evaluated"],
         "hours_evaluated": result["hours_evaluated"]}
        for result in summary
    ]).to_csv(output_dir / "summary.csv", index=False)
    all_forecasts = pd.concat(forecasts, ignore_index=True)
    lead_rows = []
    for label, group in all_forecasts.groupby("model", sort=False):
        errors = group["forecast"].to_numpy() - group["actual"].to_numpy()
        group = group.assign(abs_error=np.abs(errors), squared_error=errors ** 2)
        for lead_hour, lead in group.groupby("lead_hour", sort=True):
            lead_rows.append({
                "model": label,
                "lead_hour": int(lead_hour),
                "MAE": float(lead["abs_error"].mean()),
                "RMSE": float(np.sqrt(lead["squared_error"].mean())),
                "hours": int(len(lead)),
            })
    pd.DataFrame(lead_rows).to_csv(output_dir / "lead_metrics.csv", index=False)
    all_forecasts.to_csv(output_dir / "predictions.csv", index=False)
    scale = summary[0]["seasonal_scale"]
    period_metrics(all_forecasts, scale).to_csv(output_dir / "period_metrics.csv", index=False)
    error_slices(all_forecasts, splits.train).to_csv(output_dir / "error_slices.csv", index=False)
    uncertainty = {
        "method": "paired moving-block bootstrap over daily forecast-origin MAE",
        "block_days": 7,
        "bootstrap_replicates": 2000,
        "seed": 42,
        "interpretation": "Negative MAE difference and positive MAE reduction favor the candidate model.",
        "comparisons": {},
    }
    if args.windows == 0:
        for candidate, reference in (
            ("timesfm_lora", "timesfm_zero_shot"),
            ("timesfm_lora", "hour_of_week_profile"),
            ("timesfm_lora", "seasonal_naive"),
        ):
            uncertainty["comparisons"][candidate + "_vs_" + reference] = paired_block_bootstrap(
                all_forecasts, candidate, reference, args.horizon
            )
    else:
        uncertainty["not_computed_reason"] = (
            "Block-bootstrap intervals require the complete, consecutive daily origins; rerun with --windows 0."
        )
    (output_dir / "uncertainty.json").write_text(json.dumps(uncertainty, indent=2) + "\n")

    first_window = all_forecasts[all_forecasts["timestamp"] <
                                 all_forecasts["timestamp"].iloc[0] + pd.Timedelta(hours=args.horizon)]
    fig, ax = plt.subplots(figsize=(12, 4))
    actual = first_window[first_window["model"] == "seasonal_naive"]
    ax.plot(actual["timestamp"], actual["actual"], label="Actual", linewidth=2)
    for label, _, _ in comparisons:
        subset = first_window[first_window["model"] == label]
        ax.plot(subset["timestamp"], subset["forecast"], label=label, linewidth=1.4)
    ax.set(title=f"{args.split.title()} forecast comparison ({args.horizon} hours)",
           ylabel="Load", xlabel="Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=160)
    print(f"Comparison saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
