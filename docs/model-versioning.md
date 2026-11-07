# Saving adapter versions

Every training run must write to a new output directory. For example:

```bash
python scripts/finetune_lora.py --data data/germany_hourly_load.csv \
  --output-dir outputs/model-versions/germany/v001 \
  --train-end 2019-01-01T00:00:00Z \
  --validation-end 2020-01-01T00:00:00Z
```

The training script refuses to write into a non-empty destination so a newer run cannot silently replace an earlier adapter. A resumed run must point `--resume-adapter` to its parent checkpoint and write to a different, empty `--output-dir`. Increment the version (`v002`, `v003`, ...) for each new run. Commit `adapter_model.safetensors`, `adapter_config.json`, `project_config.json`, and a per-version README recording data provenance, split dates, command/configuration, runtime, validation history, selected checkpoint, evaluation results, and the adapter SHA-256.

Use `outputs/model-versions/registry.json` for a machine-readable index and update [the progression report](../outputs/model-versions/progression.md). The report should distinguish learned LoRA tensors from the frozen TimesFM base weights. Never check in the full base model checkpoint. The PJME adapter remains at its original path for backwards compatibility.

Current archive status and experiment metrics are in the [progression report](../outputs/model-versions/progression.md). The Germany v001 result is evaluated, but its weight archive still needs retrieval from the Kaggle results ZIP before the version is complete in Git.
