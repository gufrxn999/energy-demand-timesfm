# TimesFM 2.5 PJME LoRA adapter (Kaggle T4)

This is the best validation checkpoint from the consistent-preprocessing PJME retraining run, completed on a Kaggle NVIDIA Tesla T4. Five epochs ran with 2,000 sampled windows per epoch; epoch 1 was selected with validation loss 0.273837. The run used 512 hours of context, a 24-hour horizon, batch size 8, LoRA rank 4 / alpha 8, and FP16 model weights.

The base checkpoint (`google/timesfm-2.5-200m-transformers`) and PJME CSV are not included. Install the dependencies in `requirements-finetune.txt`, make the base model available, then pass this directory to `scripts/compare_models.py` with `--adapter outputs/timesfm-lora-pjme-t4`.

The matching full 908-window holdout metrics and comparison artifacts are in [`outputs/model-comparison-kaggle-full`](../model-comparison-kaggle-full/README.md). Row-level predictions are excluded from Git.
