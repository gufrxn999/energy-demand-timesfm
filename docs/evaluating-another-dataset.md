# Evaluating another hourly demand dataset

`scripts/compare_models.py` compares the weekly seasonal-naive baseline, TimesFM zero-shot, and the PJME LoRA adapter on identical forecast origins from a new hourly series. This is an inference evaluation: it does not fine-tune the adapter on the new dataset.

## Prepare the CSV

Create one CSV with these required columns:

```csv
timestamp,load
2024-01-01 00:00:00,123.4
2024-01-01 01:00:00,120.7
```

Use one demand series, one row per timestamp, numeric load values, and a consistent hourly timezone convention. Keep the original unit and document it with the results. The loader sorts timestamps, averages duplicate timestamps, regularizes to hourly frequency, and forward-fills gaps of at most three hours. It rejects unresolved longer gaps. Do not combine different meters or units in one series.

## Run the comparison

Install dependencies and make the TimesFM base checkpoint available locally or through Hugging Face access. Then evaluate on validation first:

```bash
pip install -r requirements-finetune.txt
python scripts/compare_models.py \
  --data path/to/hourly_demand.csv \
  --adapter outputs/timesfm-lora-causal \
  --split validation \
  --windows 0 \
  --inference-batch-size 32 \
  --output-dir outputs/other-dataset-validation
```

After using validation for any model or setting choices, run the final held-out comparison once:

```bash
python scripts/compare_models.py \
  --data path/to/hourly_demand.csv \
  --adapter outputs/timesfm-lora-causal \
  --split test \
  --windows 0 \
  --inference-batch-size 32 \
  --output-dir outputs/other-dataset-test
```

`--windows 0` evaluates every daily forecast origin in the selected chronological split. All three models are checked against the same timestamps and actual values. Each forecast uses 512 hours of history and predicts the next 24 hours by default. The split is 70% train, 15% validation, and 15% test in timestamp order; recent observations before a test origin can be used as past context, as they would be in real forecasting.

## Interpret results fairly

- Compare adapter and zero-shot TimesFM **within each dataset**, against the same seasonal-naive baseline. MAE and RMSE keep the dataset's load unit, so their absolute values are not directly comparable between datasets with different units or scales.
- MASE is scaled using the training portion of that dataset. Report the dataset, units, timezone, source/version, split, horizon, context length, evaluated window count, and metrics together.
- The PJME adapter measures transfer from PJME. A result on another series does not establish generalization to all regions or time periods.
- Do not use the test metrics to choose settings. If the adapter needs to be fine-tuned on the new dataset, treat that as a separate experiment and preserve an untouched future holdout.

The output directory contains `metrics.json`, `summary.csv`, `lead_metrics.csv`, `predictions.csv`, and `comparison.png`.
