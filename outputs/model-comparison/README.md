# PJME held-out comparison

Models were compared on the same 90 evenly spaced, non-overlapping 24-hour forecast windows from the chronological test split. The sampled forecast timestamps span 2016-02-06 08:00 through 2018-08-02 07:00 and cover 2,160 hours. The PJME CSV and per-hour prediction table are not committed.

This is a broad but sparse sample of the test period, not an all-hours evaluation. Treat the scores as preliminary; run `--windows 0` for every daily origin when a faster inference environment is available.

| Model | MAE | RMSE | MASE | MAE skill vs seasonal naive |
| --- | ---: | ---: | ---: | ---: |
| Weekly seasonal naive | 3219.16 | 4566.66 | 1.497 | 0.0% |
| TimesFM 2.5 zero-shot | 1594.68 | 2227.03 | 0.742 | 50.5% |
| TimesFM 2.5 LoRA (best checkpoint, epoch 2) | **1485.24** | **2053.16** | **0.691** | **53.9%** |

On this sample, the adapter reduced MAE by 6.9% relative to zero-shot TimesFM. Do not generalize this estimate to the full test period until the dense evaluation is completed.

The adapter had lower per-day MAE than zero-shot TimesFM on 55 of the 90 sampled forecast windows and lower MAE at 20 of 24 forecast lead hours. `lead_metrics.csv` breaks MAE and RMSE out by lead hour.

Reproduce with:

```bash
python scripts/compare_models.py \
  --data data/pjme_hourly.csv \
  --adapter outputs/timesfm-lora-cpu \
  --split test \
  --windows 90
```
