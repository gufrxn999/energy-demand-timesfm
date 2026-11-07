# PJME held-out comparison (causal preprocessing)

Models were compared on the same 90 evenly spaced, non-overlapping 24-hour forecast windows from the chronological test split. All models use causal forward-fill for up to three consecutive hours; unresolved longer gaps raise an error. The LoRA model uses the adapter in `outputs/timesfm-lora-causal`, selected at epoch 2. The base checkpoint, PJME CSV, and per-hour prediction table are not committed.

This is a broad but sparse sample of the test period, not an all-hours evaluation. The origins span 2016-02-06 08:00 through 2018-08-01 08:00 and cover 2,160 target hours through 2018-08-02 07:00.

| Model | MAE | RMSE | MASE | MAE skill vs seasonal naive |
| --- | ---: | ---: | ---: | ---: |
| Weekly seasonal naive | 3219.16 | 4566.66 | 1.497 | 0.0% |
| TimesFM 2.5 zero-shot | 1594.68 | 2227.03 | 0.742 | 50.5% |
| TimesFM 2.5 LoRA, causal loader | **1485.43** | **2053.29** | **0.691** | **53.9%** |

On these windows, the adapter reduced MAE by 6.9% relative to zero-shot TimesFM, had lower MAE on 55 of 90 forecast windows, and lower MAE on 20 of 24 lead hours. These sampled results do not establish performance across the full test period.

Reproduce with a local copy of the base checkpoint:

```bash
python scripts/compare_models.py \
  --data data/pjme_hourly.csv \
  --model-id work_hf_cache_cpu/manual_timesfm25 \
  --adapter outputs/timesfm-lora-causal \
  --split test \
  --windows 90 \
  --output-dir outputs/model-comparison-causal
```
