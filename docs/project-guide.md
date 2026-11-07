# Project guide

## What this project asks

This project tests whether adapting Google's pretrained TimesFM 2.5 model with a small LoRA adapter improves 24-hour-ahead forecasts of PJME hourly electricity demand.

It compares three approaches on matching forecast timestamps:

1. **Weekly seasonal naive:** repeat the previous week's hourly pattern.
2. **TimesFM 2.5 zero-shot:** use the pretrained model without adapting it to PJME.
3. **TimesFM 2.5 + LoRA:** use the pretrained model with an adapter trained on PJME.

The repository contains an end-to-end research prototype, an earlier CPU training run, and the latest full PJME holdout comparison from a Kaggle T4 run. The latest adapter, aggregate metrics, lead-hour metrics, and plot are committed; row-level predictions are excluded from Git.

## Start here

- [README](../README.md): overview, headline result, and quick start.
- [Results and limitations](results-and-limitations.md): metric meanings, what the test supports, caveats, and next steps.
- [Experiment log](experiment-log.md): dataset split, environment, training history, attempted runs, and timings.
- [Another-dataset evaluation guide](evaluating-another-dataset.md): CSV format and leakage-safe transfer evaluation.
- [Full Kaggle comparison](../outputs/model-comparison-kaggle-full/README.md): all 908 PJME test-window metrics.
- [T4 adapter](../outputs/timesfm-lora-pjme-t4/README.md): selected checkpoint from the full retraining run.
- [Comparison artifacts](../outputs/model-comparison/README.md): aggregate scores, plot, and lead-hour metrics.
- [Adapter notes](../outputs/timesfm-lora-cpu/README.md): training configuration and selected checkpoint.

## Repository map

```text
.
├── README.md                       # overview and quick start
├── docs/
│   ├── project-guide.md            # navigation and workflow
│   ├── results-and-limitations.md  # results explained and next steps
│   └── experiment-log.md           # detailed run history
├── src/
│   ├── data.py                     # hourly data loading and chronological split
│   ├── metrics.py                  # MAE, RMSE, and MASE
│   ├── models.py                   # baseline and TimesFM adapters
│   └── run_experiment.py           # single-model forecasting
├── scripts/
│   ├── make_demo_data.py           # synthetic smoke-test data
│   ├── finetune_lora.py            # TimesFM LoRA fine-tuning
│   └── compare_models.py           # shared-origin model comparison
├── tests/                          # unit tests
└── outputs/
    ├── timesfm-lora-cpu/           # selected adapter and training notes
    └── model-comparison/           # metrics, plot, and comparison notes
```

The normalized PJME CSV and row-level prediction table are not committed. The earlier and latest adapters and compact comparison artifacts are committed.

## Workflow

1. Prepare hourly data with timestamp and load columns. Keep local data under data/; data files are ignored by Git.
2. Use a chronological train, validation, and test split. The committed experiment uses 70%, 15%, and 15% respectively.
3. Train LoRA on training windows and select the checkpoint using validation loss.
4. Compare all models on the same forecast origins. Use validation while making choices and reserve test for final reporting.
5. Save aggregate metrics, lead-hour metrics, and a plot; record preprocessing, hardware, and sample size in the experiment log.

The latest run used 512 hours of context, a 24-hour horizon, 2,000 sampled windows per epoch, batch size 8, and LoRA rank 4 / alpha 8. Five epochs completed; epoch 1 was selected by validation loss.

## Reproduce the earlier committed CPU sample

Place the normalized series at data/pjme_hourly.csv and install the fine-tuning dependencies. Download or cache the TimesFM checkpoint if needed. Then run:

```bash
pip install -r requirements-finetune.txt
python scripts/compare_models.py \
  --data data/pjme_hourly.csv \
  --adapter outputs/timesfm-lora-cpu \
  --split test \
  --windows 90 \
  --inference-batch-size 8
```

The output may vary with dependency versions, checkpoint revision, hardware, and preprocessing. Review [Results and limitations](results-and-limitations.md) before treating a rerun as directly comparable.

## Scope

This is a learning and evaluation project, not a production forecasting service. It does not establish that fine-tuning reliably improves every test day, other regions, future periods, or operational outcomes.
