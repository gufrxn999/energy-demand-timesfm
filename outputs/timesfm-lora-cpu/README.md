# TimesFM 2.5 PJME LoRA checkpoint

This is a **partial checkpoint** from one completed epoch of LoRA fine-tuning on the PJME hourly electricity-demand dataset. The requested five-epoch run was stopped during epoch 2; do not treat this adapter as the completed training run.

- Base model: `google/timesfm-2.5-200m-transformers`
- Context: 512 hourly observations
- Forecast horizon: 24 hours
- LoRA rank: 4
- Dataset: PJME hourly energy demand; the CSV is not included in this repository.

Load it with PEFT after loading the base model:

```python
from peft import PeftModel
from transformers import TimesFm2_5ModelForPrediction

model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers"
)
model = PeftModel.from_pretrained(model, "outputs/timesfm-lora-cpu")
```
