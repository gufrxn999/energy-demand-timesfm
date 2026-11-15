# Data and artifact availability

This page records which inputs and run artifacts are available in the repository and what a rerun still needs.

## PJME input data

The project expects a normalized CSV with exactly these required columns:

```csv
timestamp,load
2002-01-01 01:00:00,30393.0
```

Use the public [Hourly Energy Consumption dataset on Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) and select `PJME_hourly.csv`. Place a normalized copy at `data/pjme_hourly.csv`. The loader accepts `timestamp` and `load`; for the Kaggle source file, rename `Datetime` to `timestamp` and `PJME_MW` to `load` before running. Preserve the original timestamps and values; the loader sorts, removes duplicate timestamps, regularizes to hourly frequency, and forward-fills gaps of at most three hours.

The normalized PJME input used by the local experiment materials is not in this repository. The `archive.zip` supplied separately in the Codex task contains a PJME CSV whose first timestamp is `2002-12-31 01:00:00`; it does not match the available normalized run input, which begins `2002-01-01 01:00:00`. Do not substitute one for the other when claiming to reproduce the committed result. The source dataset's redistribution terms were not verified, so the data file is intentionally not checked into this repository.

## Germany input data

The Germany series is OPSD national actual load in MW and UTC. Its input CSV is not committed. The [Germany transfer study](germany-transfer-study.md) records the source, period boundaries, and preprocessing details needed to obtain and prepare it. No Germany adapter weights are currently archived; that experiment is planned for a fresh Kaggle run.

## Available PJME artifacts

- The PJME T4 LoRA adapter is committed under [`outputs/timesfm-lora-pjme-t4`](../outputs/timesfm-lora-pjme-t4/README.md). The pretrained TimesFM base checkpoint is a separate Hugging Face download.
- Full PJME aggregate metrics, per-lead metrics, plot, and row-level forecasts are committed under [`outputs/model-comparison-kaggle-full`](../outputs/model-comparison-kaggle-full/README.md). Predictions are one row per model and forecast hour.
- Training and evaluation source code, dependency files, and a synthetic-data smoke test are in the repository.

## Kaggle materials and unavailable items

The historical [Germany transfer notebook](../notebooks/germany-electricity-demand-transfer-timesfm-2-5.ipynb) is included. The original PJME Kaggle notebook and its exact hosted notebook URL/export were not available during this inventory. A PJME training bundle ZIP exists in a local project copy; it duplicates repository code and includes an unverified data copy, so it was not added as a second source tree. The missing Germany adapter weights are not present in the accessible local artifacts.

To reproduce a run, obtain the input data from its source, normalize it as described above, install the fine-tuning dependencies, and follow the commands in the repository README and experiment log. Results can vary with checkpoint revision, dependency versions, hardware, and preprocessing.
