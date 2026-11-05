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

Fine-tuning stops after two validation epochs without improvement by default. To continue from an adapter, pass `--resume-adapter` with the adapter directory and keep `--output-dir` pointed to that directory. The adapter weights are restored; the optimizer state is initialized again.

The generated data is synthetic and only checks the pipeline. Use real hourly load data with `timestamp` and `load` columns for meaningful results. Training samples come only from the chronological training split, validation selects the saved adapter, and test is held out for final metrics.

## Compare models

See [the experiment log](docs/experiment-log.md) for training history, CPU/GPU feasibility, sampled held-out results, and the limits of the current benchmark.

The evaluator defaults to the validation split and uses daily, non-overlapping forecast windows. Use validation while making modeling choices. After the setup is fixed, run one final comparison on the held-out test split:

```bash
python scripts/compare_models.py \
  --data data/pjme_hourly.csv \
  --adapter outputs/timesfm-lora-cpu \
  --split test \
  --windows 90
```

This compares weekly seasonal-naive, zero-shot TimesFM, and the LoRA adapter on the same forecast timestamps. It writes aggregate MAE, RMSE, MASE, skill versus seasonal-naive, per-lead-hour MAE/RMSE, predictions, and a comparison plot under `outputs/model-comparison`. The committed report uses 90 evenly sampled daily origins. `--windows 0` evaluates every daily origin; a positive value evenly samples that many daily origins. Pass `--model-id` with a local checkpoint path when the Hugging Face checkpoint is already downloaded.

Hourly gaps of up to three hours are filled with the last observed load, a causal fill that does not use future values. Longer unresolved gaps raise an error so missing spans are not silently removed from the hourly timeline.

TimesFM 2.5 fine-tuning uses LoRA with Hugging Face Transformers and PEFT. See [the upstream example](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning). Check the model checkpoint license and hardware requirements before deployment.
