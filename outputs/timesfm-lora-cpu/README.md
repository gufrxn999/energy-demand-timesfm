# TimesFM 2.5 PJME LoRA adapter

Five epochs of LoRA fine-tuning completed on the PJME hourly electricity-demand dataset. The adapter saved here is the best validation checkpoint, selected from the five completed epochs. The PJME CSV is not included.

## Training setup

- Base model: `google/timesfm-2.5-200m-transformers`
- Context: 512 hours
- Forecast horizon: 24 hours
- Training windows: 2,000 per epoch
- Batch size: 8
- LoRA rank: 4, alpha: 8
- Learning rate: `5e-5`
- Training device: CPU
- Data split: chronological, with the final 15% reserved for validation

## Validation loss

| Completed epoch | Validation loss |
| ---: | ---: |
| 1 (resume baseline) | 0.273803 |
| 2 (selected checkpoint) | **0.272994** |
| 3 | 0.275715 |
| 4 | 0.290761 |
| 5 | 0.298793 |

The best checkpoint is from epoch 2. Later epochs completed but had higher validation loss.

## Load the adapter

```python
from peft import PeftModel
from transformers import TimesFm2_5ModelForPrediction

model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers"
)
model = PeftModel.from_pretrained(model, "outputs/timesfm-lora-cpu")
```
