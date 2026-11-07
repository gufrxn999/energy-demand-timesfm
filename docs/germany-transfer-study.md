# Germany hourly demand transfer study

## Summary

This second-dataset experiment tested whether the PJME-trained TimesFM 2.5 LoRA adapter transfers to a different hourly demand series, and whether fine-tuning a separate adapter on Germany improves performance. All models used the same daily forecast origins, 512-hour context and 24-hour horizon. Metrics are in MW except MASE. The 2020 test period was untouched during model and checkpoint selection.

## Data provenance and preparation

The series is DE_load_actual_entsoe_transparency from Open Power System Data's Time series package, release **2020-10-06**, sourced from ENTSO-E Transparency Platform data. It is Germany's national actual electricity load in **MW**, a single demand series (not combined meters or mixed regions). Timestamps are represented in UTC, which avoids repeated or missing local-clock hours across CET/CEST transitions.

- [OPSD release and documentation](https://data.open-power-system-data.org/time_series/2020-10-06/)
- [Dataset DOI](https://doi.org/10.25832/time_series/2020-10-06)
- [Source CSV](https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv)
- [Kaggle notebook and result bundle](https://www.kaggle.com/code/syedgufranahmed/germany-electricity-demand-transfer-timesfm-2-5/edit)

Selected UTC range: 2015-01-01 00:00 through 2020-09-30 23:00 (50,400 hourly records). Preparation followed the project conventions: parse/order timestamps, keep the last duplicate, hourly-mean aggregation, and causal forward-fill up to three hours. Inspection reported no duplicates, missing/invalid load values, or unresolved hourly gaps in the selected range; therefore no fill was needed. The Kaggle notebook packages the prepared timestamp,load CSV, metrics, predictions, plots, and selected Germany adapter in its downloadable ZIP.

## Fixed chronological splits and evaluation

| Partition | UTC interval (start inclusive, end exclusive) | Hours |
|---|---|---:|
| Train | 2015-01-01 to 2019-01-01 | 35,064 |
| Validation | 2019-01-01 to 2020-01-01 | 8,760 |
| Test | 2020-01-01 to 2020-10-01 | 6,576 |

Forecast context length is 512 hours and horizon is 24 hours. Origins are non-overlapping daily windows. Validation covers 365 origins; test covers 274 origins. MASE uses the training-only mean absolute 24-hour seasonal difference (4,509.2887 MW). MAE, RMSE, and MASE are computed from the same target hours for each compared model.

## Validation comparison (2019)

| Model | MAE (MW) | RMSE (MW) | MASE | MAE skill vs weekly naive |
|---|---:|---:|---:|---:|
| Weekly seasonal naive | 2,558.15 | 4,464.97 | 0.5673 | 0.0% |
| TimesFM 2.5 zero-shot | 3,217.87 | 4,737.20 | 0.7136 | -25.8% |
| PJME LoRA adapter, unchanged | 1,926.30 | 2,997.08 | 0.4272 | +24.7% |

On validation, the PJME adapter transferred positively to Germany; zero-shot TimesFM underperformed the weekly baseline.

## Germany fine-tuning

Base model: google/timesfm-2.5-200m-transformers. The Germany-specific adapter trained only on the 2015-2018 training period. Settings: LoRA rank 4, alpha 8, dropout 0.05, learning rate 5e-5, batch size 8, 2,000 sampled windows per epoch, seed 42, up to five epochs with patience 2. The 2019 validation loss selected epoch 5 (best loss 0.061289).

| Epoch | Validation loss |
|---:|---:|
| 1 | 0.072297 |
| 2 | 0.066109 |
| 3 | 0.065613 |
| 4 | 0.063866 |
| 5 | 0.061289 |

Training ran in Kaggle on one Tesla T4 (CUDA device 0); runtime exposed two T4 devices. Environment: Python 3.13.16, PyTorch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2, torchao 0.18.0. Kaggle initially provided torchao 0.10.0, below PEFT's requirement; upgrading to 0.18.0 resolved the adapter-loading error. The training command and date boundaries are recorded in the [Kaggle notebook](https://www.kaggle.com/code/syedgufranahmed/germany-electricity-demand-transfer-timesfm-2-5/edit).

## Untouched test comparison (2020-01-01 through 2020-09-30 UTC)

| Model | MAE (MW) | RMSE (MW) | MASE | MAE skill vs weekly naive |
|---|---:|---:|---:|---:|
| Weekly seasonal naive | 2,226.51 | 3,777.58 | 0.4938 | 0.0% |
| TimesFM 2.5 zero-shot | 2,950.22 | 4,395.39 | 0.6543 | -32.5% |
| PJME LoRA adapter, unchanged | 1,750.15 | 2,722.11 | 0.3881 | +21.4% |
| Germany LoRA adapter | **1,043.04** | **1,549.25** | **0.2313** | **+53.2%** |

Germany fine-tuning reduced MAE by 40.4% versus the unchanged PJME adapter and 53.2% versus the weekly baseline. The Germany adapter had lower daily MAE than the weekly baseline on 81.8% of test days, than zero-shot on 85.8%, and than the PJME adapter on 74.8%.

## Error review and limitations

The Germany adapter's median daily MAE was 784.0 MW and its 90th percentile was 1,771.0 MW. Its five worst days accounted for about 7.9% of summed daily MAE, so aggregate improvement was not limited to a few days. Highest-error dates included 2020-01-09 (5,026.6 MW), 2020-06-11 (4,874.1 MW), and 2020-01-02 (4,384.7 MW). The worst lead was forecast hour 13 at 1,344.7 MW MAE.

It still had notable regressions versus weekly naive, led by 2020-05-04 (+2,934.7 MW), 2020-04-14 (+1,338.7 MW), and 2020-06-22 (+970.6 MW). The detailed [day-level review](../outputs/germany_test_error_analysis.md) contains the additional dates. Error review was descriptive and did not influence model selection.

The result supports this Germany series and this 2020 holdout only. Two datasets do not establish broad generalization or production readiness. Raw MAE comparisons across PJME and Germany are not meaningful when units or scale differ; comparisons here are within Germany. Combined PJME+Germany adapter training was not run. To justify it, its performance would need to be checked independently against the separate adapters and baselines on both datasets.

## Interview summary

- **Why TimesFM and LoRA:** use a pretrained forecasting model and adapt a small parameter subset with a practical training footprint.
- **Why chronological splits:** forecast future observations from past observations; train on 2015-2018, select with 2019, and reserve 2020 for final evaluation.
- **What the baseline shows:** weekly seasonal naive is a simple seasonal reference that tests whether a model adds value beyond repeating the weekly pattern.
- **What the metrics mean:** MAE is average absolute error; RMSE weighs larger errors more; MASE scales against a training-only seasonal-naive error.
- **What transferred:** the PJME adapter beat the weekly baseline on Germany validation and test, while zero-shot TimesFM trailed it.
- **What fine-tuning changed:** the Germany adapter further reduced Germany test MAE, while still losing to the weekly baseline on some days.
- **Before deployment:** monitor error and bias by horizon, season, holiday, and demand regime; watch data quality and drift; define retraining and fallback rules; evaluate prospectively.
