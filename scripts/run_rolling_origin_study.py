"""Fine-tune and compare a separate adapter in each expanding PJME time fold.

Run on a GPU machine; this intentionally performs full adapter training and
full-origin inference for each fold.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_FOLDS = [
    {"name": "test_2011", "train_end": "2010-01-01",
     "validation_end": "2011-01-01", "test_end": "2012-01-01"},
    {"name": "test_2012", "train_end": "2011-01-01",
     "validation_end": "2012-01-01", "test_end": "2013-01-01"},
    {"name": "test_2013", "train_end": "2012-01-01",
     "validation_end": "2013-01-01", "test_end": "2014-01-01"},
]


def _run(command: list[str], dry_run: bool) -> None:
    print(json.dumps(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPOSITORY, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pjme_hourly.csv")
    parser.add_argument("--model-id", default="google/timesfm-2.5-200m-transformers")
    parser.add_argument("--output-dir", default="outputs/rolling-origin-pjme-v001")
    parser.add_argument("--folds-json", help="Optional JSON list of name/train_end/validation_end/test_end objects")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--horizon-len", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all fold commands without training or writing files")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = REPOSITORY / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"PJME CSV not found: {data_path}")
    folds = json.loads(Path(args.folds_json).read_text()) if args.folds_json else DEFAULT_FOLDS
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("At least two chronological folds are required")
    required = {"name", "train_end", "validation_end", "test_end"}
    seen_names = set()
    for fold in folds:
        if not required.issubset(fold):
            raise ValueError(f"Each fold must define {sorted(required)}")
        if fold["name"] in seen_names:
            raise ValueError(f"Duplicate fold name: {fold['name']}")
        seen_names.add(fold["name"])
        if not (pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["validation_end"])
                < pd.Timestamp(fold["test_end"])):
            raise ValueError(f"Fold boundaries must be strictly increasing: {fold['name']}")
    for previous, current in zip(folds, folds[1:]):
        if pd.Timestamp(current["train_end"]) < pd.Timestamp(previous["train_end"]):
            raise ValueError("Folds must use monotonically expanding training windows")
        if pd.Timestamp(current["validation_end"]) < pd.Timestamp(previous["validation_end"]):
            raise ValueError("Fold validation periods must move forward in time")
        if pd.Timestamp(current["validation_end"]) < pd.Timestamp(previous["test_end"]):
            raise ValueError("Fold test periods must not overlap")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY / output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.dry_run:
        raise FileExistsError(
            f"Study output already exists: {output_dir}. Choose a new versioned --output-dir."
        )
    if args.dry_run:
        print(json.dumps({"data": data_path.name, "folds": folds,
                          "output_dir": output_dir.name}, indent=2))

    fold_rows = []
    for fold in folds:
        fold_dir = output_dir / fold["name"]
        adapter_dir = fold_dir / "adapter"
        comparison_dir = fold_dir / "comparison"
        boundaries = ["--train-end", fold["train_end"],
                      "--validation-end", fold["validation_end"],
                      "--test-end", fold["test_end"]]
        train_command = [
            sys.executable, str(REPOSITORY / "scripts" / "finetune_lora.py"),
            "--data", str(data_path), "--model-id", args.model_id,
            *boundaries, "--context-len", str(args.context_len),
            "--horizon-len", str(args.horizon_len), "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--samples", str(args.samples),
            "--lr", str(args.lr), "--lora-r", str(args.lora_r),
            "--lora-alpha", str(args.lora_alpha), "--seed", str(args.seed),
            "--patience", str(args.patience), "--output-dir", str(adapter_dir),
        ]
        compare_command = [
            sys.executable, str(REPOSITORY / "scripts" / "compare_models.py"),
            "--data", str(data_path), "--adapter", str(adapter_dir),
            "--model-id", args.model_id, "--split", "test", *boundaries,
            "--horizon", str(args.horizon_len), "--context-length", str(args.context_len),
            "--windows", "0", "--inference-batch-size", str(args.inference_batch_size),
            "--output-dir", str(comparison_dir),
        ]
        print(f"\n=== {fold['name']} ===", flush=True)
        _run(train_command, args.dry_run)
        _run(compare_command, args.dry_run)
        if not args.dry_run:
            scores = pd.read_csv(comparison_dir / "summary.csv")
            scores["fold"] = fold["name"]
            scores["test_start"] = fold["validation_end"]
            scores["test_end_exclusive"] = fold["test_end"]
            fold_rows.append(scores)

    if args.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    all_scores = pd.concat(fold_rows, ignore_index=True)
    pivot = all_scores.pivot(index="fold", columns="model", values="MAE")
    all_scores["MAE_reduction_vs_zero_shot_pct"] = all_scores.apply(
        lambda row: 100 * (1 - row["MAE"] / pivot.loc[row["fold"], "timesfm_zero_shot"]),
        axis=1,
    )
    all_scores.to_csv(output_dir / "fold_metrics.csv", index=False)
    aggregate = all_scores.groupby("model").agg(
        folds=("fold", "nunique"),
        mean_fold_MAE=("MAE", "mean"),
        std_fold_MAE=("MAE", "std"),
        mean_fold_RMSE=("RMSE", "mean"),
        mean_fold_MASE=("MASE", "mean"),
        mean_MAE_reduction_vs_zero_shot_pct=("MAE_reduction_vs_zero_shot_pct", "mean"),
    ).reset_index()
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    study = {
        "study": "expanding-window PJME adapter evaluation",
        "model_id": args.model_id,
        "folds": folds,
        "training": {"epochs": args.epochs, "samples_per_epoch": args.samples,
                     "batch_size": args.batch_size, "seed": args.seed,
                     "selected_checkpoint": "lowest validation loss per fold"},
        "interpretation": "Each fold uses an expanding train period and its own adapter, followed by adjacent one-year validation and test periods. Test years do not overlap across folds.",
    }
    (output_dir / "study_config.json").write_text(json.dumps(study, indent=2) + "\n")
    print(f"Fold summary saved under {output_dir}")


if __name__ == "__main__":
    main()
