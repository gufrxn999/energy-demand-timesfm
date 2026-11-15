# Model progression report

## What changes after training

TimesFM 2.5 remains the shared frozen base. Each trained version adds a small PEFT/LoRA adapter; the base model's roughly 232.7 million parameters are not copied into this repository. The Germany run trained 1,382,912 adapter parameters (0.5944% of the base model parameter count). The adapter tensors are the learned weights that change model behavior.

## Saved adapter versions and evaluation progression

| Version | Training data | Selected checkpoint | Test MAE | Test RMSE | Test MASE | Weight archive |
|---|---|---:|---:|---:|---:|---|
| PJME T4 adapter | PJME training period | Epoch 1 | 1,536.45 | 2,132.71 | 0.714 | [Committed PJME adapter](../timesfm-lora-pjme-t4/README.md) |
| Germany adapter v001 | OPSD German national actual load, 2015–2018 | Epoch 5 (validation loss 0.061289) | 1,043.04 MW | 1,549.25 MW | 0.2313 | **Pending retrieval from Kaggle ZIP** |

The table reports metrics within each dataset. PJME and Germany MAE values are not comparable because they use different series and scales. On Germany's untouched 2020 period, the Germany adapter's MAE was 40.4% lower than the unchanged PJME adapter's and 53.2% lower than the weekly seasonal-naive baseline's. It had lower daily MAE than the weekly baseline on 81.8% of days; some days were worse.

The PJME adapter file SHA-256 is `4a50120f7e600a4a2b54940e273436642a27f5ebf449fe84deb461d43bdb97b8`. Its training CSV SHA-256 is `2f4f64096ea7ac5db5e02b361a88ec9ac26cf77bdfeebefb7b7b259d8821849c`. The base model ID is recorded, but the original upstream checkpoint revision was not pinned.

## Training progression for Germany v001

| Epoch | Validation loss | Change from prior epoch |
|---:|---:|---:|
| 1 | 0.072297 | — |
| 2 | 0.066109 | -8.56% |
| 3 | 0.065613 | -0.75% |
| 4 | 0.063866 | -2.66% |
| 5 | 0.061289 | -4.04% |

The validation loss fell at each epoch, so epoch 5 was selected. Training used LoRA rank 4, alpha 8, dropout 0.05, learning rate 5e-5, batch size 8, 2,000 sampled windows per epoch, seed 42, and patience 2. Training ran on a Kaggle NVIDIA T4 with PyTorch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2, and torchao 0.18.0.

## Weight-level diffs

Adapters trained independently on different datasets are separate parameterizations, not successive checkpoints in one training chain. A raw tensor subtraction between PJME and Germany LoRA weights would not explain performance and should not be presented as a meaningful “change.” For same-series continuation experiments, preserve both directories and report matching tensor keys/shapes, changed tensor count, L2 delta, and relative L2 delta alongside validation/test metrics. Record the file SHA-256 so each archived set of weights is identifiable.

## Archive status

The PJME adapter weights are committed. The Germany v001 metrics and training history are documented, but its `adapter_model.safetensors` is **not yet in Git**: Kaggle generated `/kaggle/working/germany-transfer-results.zip` (6,317,559 bytes), but the notebook's Jupyter proxy download URL returned HTTP 404 when opened from Brave. Do not treat Germany v001 as an archived/reproducible model until its adapter files and hash are added under `outputs/model-versions/germany/v001/`. The notebook continues to report the completed run; the ZIP contains the selected adapter, predictions, metrics, figures, data CSV, and manifest.
