# TimesFM 2.5 PJME LoRA adapter (causal preprocessing)

Five training epochs were completed using the leakage-safe causal forward-fill loader. The adapter saved here is the best validation checkpoint, selected from epoch 2. The PJME CSV and base checkpoint are not included.

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
- Missing-hour handling: forward-fill up to three consecutive hours; longer gaps raise an error

## Validation loss

| Completed epoch | Validation loss |
| ---: | ---: |
| 1 (resume baseline) | 0.273833 |
| 2 (selected checkpoint) | **0.273054** |
| 3 | 0.275832 |
| 4 | 0.290936 |
| 5 | 0.299268 |

The run resumed from the selected epoch 1 checkpoint after an initial attempt reached early stopping. The optimizer was reinitialized on resume. See [the experiment log](../../docs/experiment-log.md) for details.

## Load the adapter

```python
from peft import PeftModel
from transformers import TimesFm2_5ModelForPrediction

model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers"
)
model = PeftModel.from_pretrained(model, "outputs/timesfm-lora-causal")
```
