# TimesFM 2.5 PJME experiment log

Last updated: 2026-11-06

This log records the CPU feasibility work, fine-tuning history, incomplete evaluation attempts, and the final completed model comparison. The comparison below is a sampled held-out test evaluation, not a complete all-hours score.

## Dataset and split

- Dataset: PJME hourly electricity demand from the downloaded `hourly-energy-consumption.zip` archive; normalized file: `data/pjme_hourly.csv` (`timestamp`, `load`). The CSV is not committed.
- Valid source rows: 145,362 unique timestamps.
- Hourly timeline after regularization: 145,392 observations, from 2002-01-01 01:00 through 2018-08-03 00:00.
- Missing observations: 30 isolated single-hour gaps.
- Split: chronological 70% train / 15% validation / 15% test.

| Partition | Hours | Start | End |
| --- | ---: | --- | --- |
| Train | 101,774 | 2002-01-01 01:00 | 2013-08-11 14:00 |
| Validation | 21,809 | 2013-08-11 15:00 | 2016-02-06 07:00 |
| Test | 21,809 | 2016-02-06 08:00 | 2018-08-03 00:00 |

The initial fine-tuning run used the earlier preprocessing implementation, which linearly interpolated short gaps. The corrected loader uses causal forward-fill for up to three consecutive hours and rejects longer unresolved gaps. The model was subsequently retrained with this loader, and the held-out comparison was repeated; see “Causal preprocessing retrain” below. The original adapter and report are retained for comparison.

## AMD GPU feasibility

The machine has an AMD Radeon RX 5500M and integrated AMD Radeon graphics. A basic `torch-directml` tensor operation with backward propagation succeeded, but TimesFM 2.5 did not complete model training on DirectML. The available DirectML PyTorch build was 2.4.1, while the installed Transformers TimesFM 2.5 implementation requires PyTorch 2.6 or later. ROCm does not support this RX 5500M Windows setup. The model training and comparison below therefore ran on CPU.

## CPU timing and training

Environment: Windows, Python 3.11.9, PyTorch 2.15.0 CPU, Transformers 5.22.1, Accelerate 1.15.0, and PEFT 0.21.2. The machine has 12 logical processors and about 16.5 GB RAM. The checkpoint was loaded from a local cache; it contains about 925 MB of weights and loaded in about 2.8 seconds once downloaded.

A LoRA training microbenchmark on PJME used context 512, horizon 24, and batch size 8; three measured batches took 2.18, 2.33, and 2.78 seconds. The training loop uses 2,000 sampled windows per epoch (250 batches), LoRA rank 4 / alpha 8, learning rate `5e-5`, AdamW weight decay `0.01`, and gradient clipping at 1.0. PEFT reported 1,382,912 trainable parameters of 232,672,192 total (0.594%).

The full five-epoch run was resumed from the saved epoch 1 adapter. Resuming restored adapter weights and reinitialized the optimizer; it was not a bit-for-bit continuation of the original process. The epoch 1 checkpoint reevaluated at validation loss `0.273803`.

| Completed epoch | Validation loss | Epoch runtime (seconds) | Checkpoint selected? |
| ---: | ---: | ---: | --- |
| 1 (resume baseline) | 0.273803 | not recorded | No |
| 2 | **0.272994** | 606.1 | Yes; best checkpoint |
| 3 | 0.275715 | 658.2 | No |
| 4 | 0.290761 | 630.1 | No |
| 5 | 0.298793 | 605.8 | No |

The four timed resumed epochs took 2,500.2 seconds (about 41.7 minutes). All five epochs completed. The saved adapter is from epoch 2 because it had the lowest validation loss. These losses are the model's training objective and are not directly comparable to the load-unit MAE/RMSE reported below.

### Corrected causal-loader training

The adapter in `outputs/timesfm-lora-causal` was trained with the current causal forward-fill loader. The local base checkpoint was `work_hf_cache_cpu/manual_timesfm25`; it is not committed. Training used the same 2,000 sampled windows per epoch, context 512, horizon 24, batch size 8, LoRA rank 4 / alpha 8, learning rate `5e-5`, and seed 42. It ran on CPU.

The first attempt stopped after epoch 3 because the script's default patience of 2 was reached. Its selected epoch 1 checkpoint was retained, then resumed for epochs 2–5 with a freshly initialized optimizer and early stopping disabled. The table records that final five-epoch selection run; the first attempt's non-selected epochs are omitted.

| Completed epoch | Validation loss | Epoch runtime (seconds) | Checkpoint selected? |
| ---: | ---: | ---: | --- |
| 1 (resume baseline) | 0.273833 | 668.0 | No |
| 2 | **0.273054** | 603.5 | Yes; best checkpoint |
| 3 | 0.275832 | 624.2 | No |
| 4 | 0.290936 | 647.1 | No |
| 5 | 0.299268 | 634.3 | No |

The selected causal-loader checkpoint is epoch 2. Its validation loss is close to, but slightly higher than, the earlier interpolation-loader checkpoint's `0.272994`.

## Held-out comparison attempts

The evaluator generates daily non-overlapping 24-hour forecast origins. Candidate origins are sampled deterministically and evenly when `--windows` is positive. Every model in the completed comparison was checked against the same timestamps and actual load values.

| Attempt | Origins / hours | Result |
| --- | ---: | --- |
| All daily origins | 908 / 21,792 | Seasonal-naive completed (MAE 3,432.28; RMSE 4,733.95; MASE 1.596). The run was stopped during zero-shot TimesFM inference before TimesFM metrics were produced. |
| 365 sampled origins | 365 / 8,760 | Seasonal-naive completed (MAE 3,442.45; RMSE 4,765.36; MASE 1.601). The run was stopped during zero-shot inference. These baseline numbers use different origins than the final run and should not be compared to it. |
| Final sampled comparison | 90 / 2,160 | All three models completed; results below. |

The final 90 forecast origins span 2016-02-06 08:00 through 2018-08-01 08:00. Their target timestamps span 2016-02-06 08:00 through 2018-08-02 07:00. This is about one forecast day every ten days across the test interval, not dense coverage of every test day.

### Final 90-origin results

| Model | MAE | RMSE | MASE | MAE skill vs weekly seasonal naive |
| --- | ---: | ---: | ---: | ---: |
| Weekly seasonal naive | 3,219.16 | 4,566.66 | 1.497 | 0.0% |
| TimesFM 2.5 zero-shot | 1,594.68 | 2,227.03 | 0.742 | 50.5% |
| TimesFM 2.5 LoRA, best epoch 2 | **1,485.24** | **2,053.16** | **0.691** | **53.9%** |

On this sample, LoRA reduced MAE by 6.9% relative to zero-shot TimesFM, had lower MAE on 55 of 90 forecast days, and had lower aggregate MAE on 20 of 24 lead hours. No confidence interval or significance test has been calculated. Do not generalize this sampled result to the complete test interval.

MASE uses the training-set mean absolute 24-hour seasonal difference as its scale: 2,150.52 load units.

### Causal preprocessing retrain comparison

The same 90 sampled test origins were reevaluated using causal forward-fill preprocessing and the corrected epoch 2 adapter. Scores are nearly unchanged from the original sampled comparison. This remains a sparse test sample, not a dense all-hours evaluation.

| Model | MAE | RMSE | MASE | MAE skill vs weekly seasonal naive |
| --- | ---: | ---: | ---: | ---: |
| Weekly seasonal naive | 3,219.16 | 4,566.66 | 1.497 | 0.0% |
| TimesFM 2.5 zero-shot | 1,594.68 | 2,227.03 | 0.742 | 50.5% |
| TimesFM 2.5 LoRA, causal loader, best epoch 2 | **1,485.43** | **2,053.29** | **0.691** | **53.9%** |

The causal-loader adapter has lower MAE than zero-shot TimesFM on 55 of 90 forecast windows and at 20 of 24 lead hours. Its MAE is about 6.9% lower than zero-shot TimesFM and differs from the earlier adapter's sampled MAE by less than 0.02%. No confidence interval or significance test has been calculated.

### Inference throughput profile

On CPU, a 16-window TimesFM zero-shot profile took 37.22 seconds at batch size 8 (2.33 seconds per window) and 36.15 seconds at batch size 16 (2.26 seconds per window). Increasing the batch size to 16 did not materially improve throughput. Extrapolating from this short profile, two 908-origin TimesFM passes (zero-shot and LoRA) would take roughly 68 minutes, excluding model loading and runtime variation. That is an estimate, not a completed full-test measurement.

## Artifacts and reproduction

- Best adapter: [`outputs/timesfm-lora-cpu/`](../outputs/timesfm-lora-cpu/)
- Aggregate comparison, plot, and lead-hour metrics: [`outputs/model-comparison/`](../outputs/model-comparison/)
- Causal-loader best adapter: [`outputs/timesfm-lora-causal/`](../outputs/timesfm-lora-causal/)
- Causal-loader comparison: [`outputs/model-comparison-causal/`](../outputs/model-comparison-causal/)
- The row-level `predictions.csv` is generated locally by the comparison script but is intentionally not committed.

Run a CPU-friendlier sampled comparison with:

```bash
python scripts/compare_models.py \
  --data data/pjme_hourly.csv \
  --adapter outputs/timesfm-lora-cpu \
  --split test \
  --windows 90 \
  --inference-batch-size 8
```

For every daily origin, set `--windows 0`; expect substantially longer CPU runtime. Use `--split validation` while making model choices, and reserve `--split test` for a final report.

Reproduce the causal-loader evaluation by supplying the local checkpoint directory:

```bash
python scripts/compare_models.py \
  --data data/pjme_hourly.csv \
  --model-id work_hf_cache_cpu/manual_timesfm25 \
  --adapter outputs/timesfm-lora-causal \
  --split test \
  --windows 90 \
  --output-dir outputs/model-comparison-causal
```
