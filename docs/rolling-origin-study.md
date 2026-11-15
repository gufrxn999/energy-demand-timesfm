# Expanding-window PJME adapter study

The committed full comparison has one chronological test interval. The new study runner measures how fine-tuned TimesFM behaves across three earlier one-year test periods, with a separate adapter selected on validation data for each fold. Run it on Kaggle or another GPU machine; it is intentionally not run during ordinary local setup because it trains and evaluates three adapters.

## Default folds

| Fold | Training ends before | Validation period | Test period |
| --- | --- | --- | --- |
| 2011 | 2010-01-01 | 2010 | 2011 |
| 2012 | 2011-01-01 | 2011 | 2012 |
| 2013 | 2012-01-01 | 2012 | 2013 |

Each fold's training window expands, and its validation and test periods are consecutive calendar years. Test periods do not overlap. A later fold can use earlier historical data, including a year that was a previous fold's test period, because that year is in the past at the later forecast cutoff.

## Run on Kaggle

Clone or upload this repository, ensure `data/pjme_hourly.csv` is present, enable a GPU, and install the fine-tuning requirements:

```bash
pip install -r requirements-finetune.txt
python scripts/run_rolling_origin_study.py --dry-run
python scripts/run_rolling_origin_study.py
```

The dry run prints the exact training and comparison commands without writing files. A full run trains each fold's adapter independently, selects its checkpoint using that fold's validation period, and evaluates all daily origins in that fold's test year. It saves the adapters and full comparisons under `outputs/rolling-origin-pjme-v001/`, plus `fold_metrics.csv`, `aggregate_metrics.csv`, and `study_config.json`.

The default is five epochs, 2,000 sampled windows per epoch, batch size 8, LoRA rank 4 / alpha 8, seed 42, 512 context hours, and a 24-hour forecast. To keep a later run separate, pass a new versioned `--output-dir`. The script refuses to overwrite a non-empty study directory.

The aggregate reports an unweighted mean across fold scores and the standard deviation of fold MAE. It complements the existing 2016–2018 holdout report; it does not replace it. Interpret the folds as related rolling evaluations, because later folds have more training history and may use earlier calendar years that appeared as previous-fold test data. Do not tune settings on these folds and report their mean as if the folds were independent replications.

## Limits

This workflow has been added but has not been run on GPU, so fold adapters and multi-fold results are not yet available. It requires a GPU session with the base TimesFM checkpoint, training dependencies, and enough runtime to train three adapters and run their full test comparisons. The original Kaggle checkpoint's exact upstream model revision was not recorded; newly trained folds record the locally resolved revision when the Hugging Face cache exposes it.
