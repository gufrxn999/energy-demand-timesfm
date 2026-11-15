# TimesFM 2.5 PJME LoRA adapter (Kaggle T4)

This folder contains the selected PJME adapter weights and the metadata needed to load and identify them. It is a PEFT/LoRA adapter for the separate base checkpoint `google/timesfm-2.5-200m-transformers`; the 200M-parameter base weights are not copied into this repository.

## Selected checkpoint

The adapter came from the consistent-preprocessing PJME retraining run on a Kaggle NVIDIA Tesla T4. Five epochs ran; epoch 1 was selected using validation loss `0.27383723435923457`. Training used 512 hours of context, a 24-hour horizon, 2,000 sampled windows per epoch, batch size 8, LoRA rank 4 / alpha 8, learning rate `5e-5`, seed 42, and FP16 weights.

The normalized training data is [`data/pjme_hourly.csv`](../../data/pjme_hourly.csv). Its SHA-256 is `2f4f64096ea7ac5db5e02b361a88ec9ac26cf77bdfeebefb7b7b259d8821849c`. The adapter weights have SHA-256 `4a50120f7e600a4a2b54940e273436642a27f5ebf449fe84deb461d43bdb97b8`; the hash is also recorded in `project_config.json` and the model registry.

The loader regularizes the source series to hourly frequency using hourly means and causal forward-fill for gaps up to three hours. The model was trained on the first 70% of this timeline; validation and test each used the next 15%. The split sizes after regularization were 101,774 train hours, 21,809 validation hours, and 21,809 test hours.

The base-model revision was not recorded during the original Kaggle run (`adapter_config.json` has no revision pin). The model can be loaded by its model ID, but reproducing the exact original base checkpoint is not guaranteed if that upstream model changes.

## Evaluate this adapter

Install the fine-tuning/inference dependencies and ensure the base checkpoint is available, then run the full test comparison:

```bash
pip install -r requirements-finetune.txt
python scripts/compare_models.py \
  --data data/pjme_hourly.csv \
  --adapter outputs/timesfm-lora-pjme-t4 \
  --split test \
  --windows 0 \
  --inference-batch-size 32
```

The reported test result covers 908 daily origins and 21,792 hourly predictions: MAE `1536.4459`, RMSE `2132.7119`, and MASE `0.714454`. The [full comparison folder](../model-comparison-kaggle-full/README.md) contains aggregate and lead-hour metrics, the plot, and row-level predictions. These metrics describe one PJME series and one chronological holdout; no confidence interval or significance test was calculated.

To train a new PJME adapter, use the [GPU training notebook](../../notebooks/timesfm-pjme-gpu-training.ipynb) or write a new version to a separate output path. Do not overwrite this selected checkpoint.
