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

## Germany transfer study

A second dataset study evaluated the OPSD Germany actual-load series (MW, UTC) with fixed chronological train, validation, and test periods. On 2019 validation, the unchanged PJME LoRA adapter beat the weekly seasonal baseline, while zero-shot TimesFM did not. On the untouched 2020 test period, a separately fine-tuned Germany adapter performed best.

| Germany test model | MAE (MW) | RMSE (MW) | MASE |
| --- | ---: | ---: | ---: |
| Weekly seasonal naive | 2,226.51 | 3,777.58 | 0.4938 |
| TimesFM 2.5 zero-shot | 2,950.22 | 4,395.39 | 0.6543 |
| PJME LoRA adapter, unchanged | 1,750.15 | 2,722.11 | 0.3881 |
| Germany LoRA adapter | **1,043.04** | **1,549.25** | **0.2313** |

The Germany adapter lowered MAE 40.4% versus the PJME adapter and 53.2% versus the weekly baseline across 274 daily origins (6,576 forecast hours). Gains were not universal across days. See the full [Germany transfer study](docs/germany-transfer-study.md), [validation metrics](outputs/germany_validation_summary.csv), [test metrics](outputs/germany_test_summary.csv), [training configuration](outputs/germany_training_config.json), and [test error analysis](outputs/germany_test_error_analysis.md). The report links the OPSD release, split boundaries, evaluation details, Kaggle notebook and limitations.

The historical Kaggle notebook used for that Germany run is now included at [`notebooks/germany-electricity-demand-transfer-timesfm-2-5.ipynb`](notebooks/germany-electricity-demand-transfer-timesfm-2-5.ipynb). It records the earlier run workflow; use it as a reference when rebuilding the Germany run, not as evidence that the missing Germany adapter has been archived.

## Model versions and training progression

The [model progression report](outputs/model-versions/progression.md) records adapter training history, checkpoints, test comparisons, and archive status. The [versioning guide](docs/model-versioning.md) explains how to preserve each future run under a separate immutable directory. The PJME adapter weights are in Git. Germany v001's training and test results are recorded, but its adapter weight file still needs to be retrieved from the Kaggle results ZIP before it is fully archived in Git.
## What is complete

- Hourly data loading, chronological splits, forecast models, and MAE/RMSE/MASE evaluation.
- Unit tests and GitHub Actions for the lightweight test suite.
- Five LoRA training epochs on PJME, run on Kaggle's NVIDIA T4. The best validation loss was 0.273837 at epoch 1; later epochs did not improve it.
- A shared-origin comparison of weekly seasonal naive, zero-shot TimesFM, and the LoRA adapter across all 908 test windows.
- A separate [dataset evaluation guide](docs/evaluating-another-dataset.md) for measuring transfer to another hourly series without fitting on its test partition.

The run used 512 hours of context, a 24-hour horizon, 2,000 sampled windows per epoch, batch size 8, and LoRA rank 4 / alpha 8. See the [experiment log](docs/experiment-log.md) and [results and limitations](docs/results-and-limitations.md) for provenance and caveats.

The trained T4 adapter is committed in [outputs/timesfm-lora-pjme-t4](outputs/timesfm-lora-pjme-t4/README.md), along with aggregate metrics, per-lead metrics, a plot, and row-level predictions in [outputs/model-comparison-kaggle-full](outputs/model-comparison-kaggle-full/README.md).

The normalized PJME input is included at [`data/pjme_hourly.csv`](data/pjme_hourly.csv), and the GPU training/export workflow is in [`notebooks/timesfm-pjme-gpu-training.ipynb`](notebooks/timesfm-pjme-gpu-training.ipynb). The full TimesFM base checkpoint is downloaded separately from Hugging Face.

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

The PJME input CSV is included; the Germany input CSV is not. See [data and artifact availability](docs/data-and-artifact-availability.md) for sources, formats, and remaining gaps. See [Project guide](docs/project-guide.md) for the repository map and workflow, and [Results and limitations](docs/results-and-limitations.md) for metric definitions, caveats, and next steps.

## Scope

This is a learning and evaluation prototype, not a production forecasting service. It does not establish reliable gains across every test day, other regions, or future operating conditions.
