# Versioned model adapters

This directory is the immutable archive for trained adapter versions. Each run belongs in its own path:

```text
outputs/model-versions/<dataset>/v001/
  adapter_config.json
  adapter_model.safetensors
  project_config.json
  README.md
```

Use a new version directory for every training run (including a resumed run). Keep the base model ID, training data and split boundaries, environment, selected epoch, validation history, metrics, and SHA-256 of `adapter_model.safetensors` in that version's README or manifest. Never commit base TimesFM weights; the adapter is loaded on top of the named Hugging Face base model.

The existing adapters in `outputs/timesfm-lora-*` predate this archive convention and remain at their original paths. See [the progression report](progression.md) and [the versioning guide](../../docs/model-versioning.md).
