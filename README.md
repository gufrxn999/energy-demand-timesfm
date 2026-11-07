# PJME Electricity Demand Forecasting with TimesFM

A research project testing whether LoRA fine-tuning of a pretrained time-series model improves 24-hour-ahead forecasts of PJME hourly electricity demand.

> **Status:** A consistent-preprocessing LoRA run and full PJME test comparison have completed on Kaggle's NVIDIA T4 GPU. The comparison covers all 908 daily test windows (21,792 hourly forecasts). The adapter improved aggregate MAE by 2.9% versus zero-shot TimesFM on this PJME holdout. This is evidence for this one series and holdout, not a deployment guarantee. See [Results and limitations](docs/results-and-limitations.md).

## Research question

Does adapting TimesFM 2.5 to PJME improve forecasts over the pretrained model and a weekly seasonal-naive baseline?

## Full PJME held-out comparison

All three models were evaluated on the same 908 daily, non-overlapping forecast origins (21,792 hours) from the chronological test partition. This run used the causal forward-fill preprocessing used by the current loader.

| Model | MAE | RMSE | MASE | MAE skill vs weekly naive |
| --- | ---: | ---: | ---: | ---: |
| Weekly seasonal naive | 3,432.28 | 4,733.95 | 1.596 | 0.0% |
| TimesFM 2.5 zero-shot | 1,582.77 | 2,214.42 | 0.736 | 53.9% |
| TimesFM 2.5 LoRA (best validation checkpoint, epoch 1) | **1,536.45** | **2,132.71** | **0.714** | **55.2%** |

Across all test windows, LoRA lowered MAE by 2.9% versus zero-shot TimesFM. No confidence interval or significance test has been calculated.

## What is complete

- Hourly data loading, chronological splits, forecast models, and MAE/RMSE/MASE evaluation.
- Unit tests and GitHub Actions for the lightweight test suite.
- Five LoRA training epochs on PJME, run on Kaggle's NVIDIA T4. The best validation loss was 0.273837 at epoch 1; later epochs did not improve it.
- A shared-origin comparison of weekly seasonal naive, zero-shot TimesFM, and the LoRA adapter across all 908 test windows.
- A separate [dataset evaluation guide](docs/evaluating-another-dataset.md) for measuring transfer to another hourly series without fitting on its test partition.

The run used 512 hours of context, a 24-hour horizon, 2,000 sampled windows per epoch, batch size 8, and LoRA rank 4 / alpha 8. See the [experiment log](docs/experiment-log.md) and [results and limitations](docs/results-and-limitations.md) for provenance and caveats.

The trained T4 adapter is committed in [outputs/timesfm-lora-pjme-t4](outputs/timesfm-lora-pjme-t4/README.md), along with aggregate metrics, per-lead metrics, and a plot in [outputs/model-comparison-kaggle-full](outputs/model-comparison-kaggle-full/README.md). Row-level predictions are excluded from Git.

## Quick start

This smoke test uses synthetic data and the seasonal-naive baseline; it does not reproduce the PJME result.


```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_demo_data.py
python -m src.run_experiment --data data/demo_hourly.csv --model seasonal_naive
```

## Reproduce the committed sample

Use the normalized PJME CSV at `data/pjme_hourly.csv` and install fine-tuning dependencies. Download/cache the TimesFM checkpoint if needed, then run:

```bash
pip install -r requirements-finetune.txt
python scripts/compare_models.py \
  --data data/pjme_hourly.csv \
  --adapter outputs/timesfm-lora-causal \
  --split test \
  --windows 90 \
  --inference-batch-size 8
```

Use `outputs/timesfm-lora-pjme-t4` for the full 908-window T4 run. The earlier CPU adapters and sampled reports are retained for comparison. See [the experiment log](docs/experiment-log.md) for run history and benchmark limitations.

The real PJME CSV and row-level predictions are not committed. See [Project guide](docs/project-guide.md) for the repository map and workflow, and [Results and limitations](docs/results-and-limitations.md) for metric definitions, caveats, and next steps.

## Scope

This is a learning and evaluation prototype, not a production forecasting service. It does not establish reliable gains across every test day, other regions, or future operating conditions.
