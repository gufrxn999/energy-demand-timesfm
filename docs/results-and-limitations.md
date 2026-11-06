# Results, interpretation, and limitations

## What has been achieved

The project has moved beyond a synthetic-data demo. A LoRA adapter was trained on real PJME hourly electricity demand, selected using validation loss, and evaluated against both zero-shot TimesFM 2.5 and a weekly seasonal-naive forecast. The adapter and compact evaluation artifacts are in the repository.

The training and final comparison ran on CPU. Attempts to use the MSI's AMD GPU did not support the required TimesFM 2.5 stack, so the experiment continued on CPU instead.

## Training run

The chronological data split contains 70% training, 15% validation, and 15% test observations. Fine-tuning used 512 hours of context, forecast 24 hours, sampled 2,000 windows per epoch, batch size 8, and LoRA rank 4 / alpha 8. Five epochs completed. The lowest recorded validation loss was **0.272994 at epoch 2**, so that adapter is the selected checkpoint. Epochs 3–5 had higher validation loss.

The full run resumed from a saved epoch 1 adapter. Adapter weights were restored, but the optimizer state was reinitialized. It should therefore be described as a resumed run, not an uninterrupted bit-for-bit continuation. Detailed epoch times and machine/software versions are in the [experiment log](experiment-log.md).

Validation loss is the model's training objective. It is not in the same units as forecast MAE or RMSE, so the two values should not be compared directly.

## Held-out comparison

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

## Why the conclusion is preliminary

1. **Sparse test coverage.** The 90 origins span almost the whole test interval, but are about one forecast day every ten days. They are not a score over every test day. A full run has 908 daily origins (21,792 forecast hours).
2. **Preprocessing mismatch.** Fine-tuning used an earlier loader that linearly interpolated short gaps. Final evaluation uses causal forward-fill for gaps up to three hours and rejects longer gaps. There are only 30 isolated single-hour gaps in the regularized series, but training and evaluation are not perfectly matched. Retrain with the current loader before treating the comparison as final.
3. **Uncertainty is unmeasured.** The 90-window improvement may be sensitive to which dates were sampled. There is no confidence interval or paired statistical test.
4. **One series and one split.** This tests one PJME series and one chronological holdout. It does not establish performance on other regions, datasets, seasons, or future deployment conditions.
5. **Data and detailed predictions are local.** The normalized PJME CSV and row-level prediction file are not committed, so a fresh clone needs the dataset before it can reproduce the evaluation.

## Earlier evaluation attempts

The all-daily-origin attempt completed the baseline but was stopped during zero-shot inference before TimesFM results were produced. A 365-window attempt was also stopped during zero-shot inference. Their baseline metrics used different origins from the final 90-window run and should not be compared with the final table. The finished 90-window run is the only committed comparison with metrics for all three models.

## Recommended next steps

1. **Retrain with the current causal gap-filling loader** so training and evaluation preprocessing match.
2. **Run the full test comparison** with --windows 0 when an environment with enough inference throughput is available. Keep all models on identical daily forecast origins.
3. **Quantify uncertainty** with paired daily errors and a time-aware block bootstrap or another suitable paired procedure. Avoid treating adjacent hourly errors as independent samples.
4. **Keep the test split untouched** while tuning. Select context, LoRA settings, and checkpoint on training/validation data; use the test set for the final report.
5. **Record provenance**: dataset source/version, checkpoint revision, preprocessing version, command, package versions, hardware, and output artifacts.

The committed sampled output files are [summary.csv](../outputs/model-comparison/summary.csv), [metrics.json](../outputs/model-comparison/metrics.json), [lead_metrics.csv](../outputs/model-comparison/lead_metrics.csv), and [comparison.png](../outputs/model-comparison/comparison.png). The full run history is in the [experiment log](experiment-log.md).
