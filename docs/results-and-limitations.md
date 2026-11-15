# Results, interpretation, and limitations

## What has been achieved

The project has moved beyond a synthetic-data demo. The latest run retrained LoRA on PJME with the current causal loader and completed the full 908-window test comparison on a Kaggle NVIDIA T4. The adapter and compact evaluation artifacts are committed. The earlier CPU adapter and sampled report remain in the repository.

The earlier experiment ran on CPU. The latest full comparison used Kaggle's NVIDIA T4 GPU; the T4 uses FP16 because it does not support BF16.

## Latest full PJME comparison

The latest retraining used the same causal forward-fill loader as evaluation, with 512 context hours, a 24-hour horizon, 2,000 sampled windows per epoch, batch size 8, and five epochs. The best validation checkpoint was epoch 1 (validation loss 0.273837). The final test comparison covers every daily test origin: 908 windows and 21,792 forecast hours. It includes weekly seasonal naive, a training-fitted hour-of-week profile, zero-shot TimesFM, and PJME LoRA.

| Model | MAE | RMSE | MASE |
| --- | ---: | ---: | ---: |
| Weekly seasonal naive | 3,432.28 | 4,733.95 | 1.596 |
| Hour-of-week profile, recent-level adjusted | 2,771.75 | 3,898.73 | 1.289 |
| TimesFM 2.5 zero-shot | 1,582.77 | 2,214.42 | 0.736 |
| TimesFM 2.5 LoRA | **1,536.45** | **2,132.71** | **0.714** |

The LoRA adapter lowered MAE by 2.93% against zero-shot TimesFM across this holdout. A paired moving-block bootstrap with seven-day blocks and 2,000 resamples gives a 95% interval of 0.64% to 5.11% for the MAE reduction. LoRA had lower MAE than zero-shot TimesFM in 8 of the 11 calendar-quarter slices, so gains were not uniform over time. The [adapter](../outputs/timesfm-lora-pjme-t4/README.md), [aggregate metrics](../outputs/model-comparison-kaggle-full/summary.csv), [quarterly metrics](../outputs/model-comparison-kaggle-full/period_metrics.csv), [bootstrap intervals](../outputs/model-comparison-kaggle-full/uncertainty.json), [per-lead metrics](../outputs/model-comparison-kaggle-full/lead_metrics.csv), [plot](../outputs/model-comparison-kaggle-full/comparison.png), and [row-level predictions](../outputs/model-comparison-kaggle-full/predictions.csv) are committed.

The hour-of-week profile is fit using training data only. It predicts the median load for each hour-of-week and scales that profile by the median actual-to-profile ratio from the latest context week. This adds an interpretable statistical comparator; it is not a fitted ARIMA model.

## Earlier CPU training run

The chronological data split contains 70% training, 15% validation, and 15% test observations. Fine-tuning used 512 hours of context, forecast 24 hours, sampled 2,000 windows per epoch, batch size 8, and LoRA rank 4 / alpha 8. Five epochs completed. The lowest recorded validation loss was **0.272994 at epoch 2**, so that adapter is the selected checkpoint. Epochs 3–5 had higher validation loss.

The full run resumed from a saved epoch 1 adapter. Adapter weights were restored, but the optimizer state was reinitialized. It should therefore be described as a resumed run, not an uninterrupted bit-for-bit continuation. Detailed epoch times and machine/software versions are in the [experiment log](experiment-log.md).

Validation loss is the model's training objective. It is not in the same units as forecast MAE or RMSE, so the two values should not be compared directly.

## Earlier 90-window comparison

The completed comparison uses 90 evenly sampled daily forecast origins in the chronological test partition. Each origin produces a non-overlapping 24-hour forecast, for 2,160 evaluated hourly predictions. All three methods use the same forecast timestamps and actual values.

| Model | MAE | RMSE | MASE | MAE skill vs weekly naive |
| --- | ---: | ---: | ---: | ---: |
| Weekly seasonal naive | 3,219.16 | 4,566.66 | 1.497 | 0.0% |
| TimesFM 2.5 zero-shot | 1,594.68 | 2,227.03 | 0.742 | 50.5% |
| TimesFM 2.5 LoRA, epoch 2 | **1,485.24** | **2,053.16** | **0.691** | **53.9%** |

### Reading the result

- **MAE** is the average absolute forecast error. Lower is better. The LoRA model's MAE is 6.9% lower than zero-shot TimesFM on these sampled windows.
- **RMSE** penalizes large errors more strongly than MAE. The LoRA model also has the lowest RMSE in this sample.
- **MASE** scales forecast error by a training-set seasonal-naive error. Values below 1 indicate lower average error than that scale. The exact scaling is described in the experiment log.
- **MAE skill vs weekly naive** is the percentage reduction in MAE against the weekly seasonal-naive model.

On the sampled forecast days, LoRA has lower daily MAE than zero-shot TimesFM on 55 of 90 days and lower aggregate MAE on 20 of 24 forecast lead hours. These are useful descriptive comparisons, but no confidence interval or significance test has been calculated.

## Limitations of the latest result

1. **The interval applies to one holdout.** The block bootstrap accounts for serial dependence between nearby daily forecast origins, but it does not establish performance on new regions or future operating conditions.
2. **One series and one chronological holdout.** Quarter slices reveal variation within the 2016–2018 test interval; they are not independent retraining/test folds.
3. **Reproduction needs the base model.** The normalized PJME CSV is included, but the TimesFM base checkpoint is not. A fresh environment needs the checkpoint and compatible inference dependencies. See [data and artifact availability](data-and-artifact-availability.md) for input details.

## Earlier evaluation attempts

Earlier sampled comparisons used fewer origins and should not be compared directly with the full test table. The current full table evaluates the same 908 daily origins for each model.

## Recommended next steps

1. **Evaluate another hourly dataset** using the [transfer evaluation guide](evaluating-another-dataset.md). Keep its units and time zone explicit and compare all models on the same future holdout.
2. **Add true multiple training/test folds** by fitting separate adapters at earlier chronological cutoffs. The quarterly breakdown and bootstrap intervals here still come from one fixed holdout.
3. **Keep each test split untouched** while tuning. Select context, LoRA settings, and checkpoint on training/validation data; use the test set for the final report.
4. **Record provenance**: dataset source/version, checkpoint revision, preprocessing version, command, package versions, hardware, and output artifacts.

The committed sampled output files are [summary.csv](../outputs/model-comparison/summary.csv), [metrics.json](../outputs/model-comparison/metrics.json), [lead_metrics.csv](../outputs/model-comparison/lead_metrics.csv), and [comparison.png](../outputs/model-comparison/comparison.png). The full run history is in the [experiment log](experiment-log.md).
