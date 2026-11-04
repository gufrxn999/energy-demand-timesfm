# Local Energy Demand Forecasting with TimesFM

A small research project testing whether adapting a pretrained time-series model improves hourly electricity demand forecasts.

## Research question
Does fine-tuning a pretrained TimesFM checkpoint on local hourly demand improve 24-hour-ahead forecasts over zero-shot TimesFM and a seasonal-naive baseline?

## Quick start
Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_demo_data.py
python -m src.run_experiment --data data/demo_hourly.csv --model seasonal_naive
```

Install the fine-tuning dependencies with `pip install -r requirements-finetune.txt`. Fine-tune and evaluate a LoRA adapter with:

```bash
python scripts/finetune_lora.py --data data/demo_hourly.csv --context-len 512 --horizon-len 24 --output-dir outputs/timesfm-lora
python -m src.run_experiment --data data/demo_hourly.csv --model timesfm --adapter outputs/timesfm-lora
```

The generated data is synthetic and only checks the pipeline. Use real hourly load data with `timestamp` and `load` columns for meaningful results. Training samples come only from the chronological training split, validation selects the saved adapter, and test is held out for final metrics.

TimesFM 2.5 fine-tuning uses LoRA with Hugging Face Transformers and PEFT. See [the upstream example](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning). Check the model checkpoint license and hardware requirements before deployment.
