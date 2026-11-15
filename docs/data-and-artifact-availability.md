# Data and artifact availability

This page records which inputs and run artifacts are available in the repository and what a rerun still needs.

## PJME input data

The exact normalized CSV used for PJME training and evaluation is included at [`data/pjme_hourly.csv`](../data/pjme_hourly.csv). Its SHA-256 is `2f4f64096ea7ac5db5e02b361a88ec9ac26cf77bdfeebefb7b7b259d8821849c`. It has 145,366 hourly observations from 2002-01-01 01:00 through 2018-08-03 00:00, in megawatts.

The source is Rob Mulla's [Hourly Energy Consumption dataset on Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption), whose dataset page identifies PJM as the data source and declares CC0. The project's normalized CSV has these required columns:

```csv
timestamp,load
2002-01-01 01:00:00,30393.0
```

The source CSV uses `Datetime` and `PJME_MW`; the included project file renames them to `timestamp` and `load`. The loader sorts, removes duplicate timestamps, regularizes to hourly frequency, and forward-fills gaps of at most three hours.

The separately supplied `archive.zip` contains a different PJME CSV whose first timestamp is `2002-12-31 01:00:00`. Use the committed normalized file above when reproducing the reported run.

## Germany input data

The Germany series is OPSD national actual load in MW and UTC. Its input CSV is not committed. The [Germany transfer study](germany-transfer-study.md) records the source, period boundaries, and preprocessing details needed to obtain and prepare it. No Germany adapter weights are currently archived; that experiment is planned for a fresh Kaggle run.

## Available PJME artifacts

- The PJME T4 LoRA adapter is committed under [`outputs/timesfm-lora-pjme-t4`](../outputs/timesfm-lora-pjme-t4/README.md). The pretrained TimesFM base checkpoint is a separate Hugging Face download.
- Full PJME aggregate metrics, per-lead metrics, plot, and row-level forecasts are committed under [`outputs/model-comparison-kaggle-full`](../outputs/model-comparison-kaggle-full/README.md). Predictions are one row per model and forecast hour.
- Training and evaluation source code, dependency files, and a synthetic-data smoke test are in the repository.

## Kaggle materials and unavailable items

The [PJME GPU training notebook](../notebooks/timesfm-pjme-gpu-training.ipynb) and historical [Germany transfer notebook](../notebooks/germany-electricity-demand-transfer-timesfm-2-5.ipynb) are included. The supplied PJME training bundle contained the exact CSV now committed here; its duplicate code was not added as a ZIP. The evaluation-code ZIP duplicates repository source, and the result ZIP's adapter and predictions are already represented in the repository. The missing Germany adapter weights are not present in the accessible local artifacts.

To reproduce a run, obtain the input data from its source, normalize it as described above, install the fine-tuning dependencies, and follow the commands in the repository README and experiment log. Results can vary with checkpoint revision, dependency versions, hardware, and preprocessing.
