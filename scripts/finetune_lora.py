"""LoRA fine-tuning for TimesFM 2.5 on a single hourly load CSV.

Training windows come only from the training partition. The final chronological
15% is held out for evaluation and is never used for adapter selection.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import chronological_split, load_hourly_csv


class WindowDataset(Dataset):
    def __init__(self, values, context_len, horizon_len, samples, seed):
        self.values = np.asarray(values, dtype=np.float32)
        self.context_len, self.horizon_len = context_len, horizon_len
        total = context_len + horizon_len
        if len(self.values) < total:
            raise ValueError(f"Training partition needs at least {total} observations")
        rng = np.random.default_rng(seed)
        self.starts = rng.integers(0, len(self.values) - total + 1, size=samples)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        split = start + self.context_len
        return (torch.from_numpy(self.values[start:split].copy()),
                torch.from_numpy(self.values[split:split + self.horizon_len].copy()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-id", default="google/timesfm-2.5-200m-transformers")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--horizon-len", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--output-dir", default="outputs/timesfm-lora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-adapter", default=None,
                         help="Resume from a PEFT adapter directory; --epochs counts additional epochs")
    parser.add_argument("--base-model-name", default=None,
                         help="Canonical base model name to store in adapter metadata")
    parser.add_argument("--patience", type=int, default=2,
                         help="Stop after this many non-improving validation epochs; zero disables")
    args = parser.parse_args()
    if args.context_len % 32:
        raise ValueError("TimesFM 2.5 context length must be a multiple of 32")
    if args.patience < 0:
        raise ValueError("patience must be zero or a positive number")

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # T4 and older NVIDIA GPUs require FP16; Ampere and newer support BF16.
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32
    split = chronological_split(load_hourly_csv(args.data))
    context_len = args.context_len
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map=device
    )
    context_len = min(context_len, model.config.context_length)
    if context_len % 32:
        context_len = context_len - context_len % 32
    train_ds = WindowDataset(split.train, context_len, args.horizon_len, args.samples, args.seed)
    val_samples = min(256, max(1, len(split.validation) - context_len - args.horizon_len + 1))
    val_ds = WindowDataset(split.validation, context_len, args.horizon_len,
                           val_samples, args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    output_dir = Path(args.output_dir)
    base_model_name = args.base_model_name or args.model_id
    starting_epoch = 0
    best_checkpoint_epoch = 0
    if args.resume_adapter:
        resume_dir = Path(args.resume_adapter)
        prior_config = json.loads((resume_dir / "project_config.json").read_text())
        starting_epoch = int(prior_config.get(
            "checkpoint_epoch", prior_config.get("epochs_completed", 0)
        ))
        best_checkpoint_epoch = starting_epoch
        model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear",
            lora_dropout=0.05, bias="none",
        ))
    model.peft_config["default"].base_model_name_or_path = base_model_name
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_val = float("inf")

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for past, future in val_loader:
                out = model(past_values=past.to(device), future_values=future.to(device),
                            forecast_context_len=context_len)
                losses.append(float(out.loss))
        return float(np.mean(losses))

    if args.resume_adapter:
        best_val = evaluate()
        print(f"resume_from_epoch={starting_epoch} baseline_validation_loss={best_val:.6f}",
              flush=True)

    completed_epochs = starting_epoch
    stale_epochs = 0
    for epoch in range(args.epochs):
        total_epoch = starting_epoch + epoch + 1
        model.train()
        epoch_start = time.perf_counter()
        interval_start = epoch_start
        for step, (past, future) in enumerate(train_loader, start=1):
            past, future = past.to(device), future.to(device)
            loss = model(past_values=past, future_values=future,
                         forecast_context_len=context_len).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 25 == 0 or step == len(train_loader):
                now = time.perf_counter()
                print(f"epoch={total_epoch}/{starting_epoch + args.epochs} "
                      f"batch={step}/{len(train_loader)} loss={float(loss.detach()):.6f} "
                      f"seconds_per_batch={(now - interval_start) / min(25, step % 25 or 25):.2f}",
                      flush=True)
                interval_start = now
        val_loss = evaluate()
        print(f"epoch={total_epoch}/{starting_epoch + args.epochs} "
              f"validation_loss={val_loss:.6f} epoch_seconds={time.perf_counter() - epoch_start:.1f}",
              flush=True)
        completed_epochs = total_epoch
        if val_loss < best_val:
            best_val = val_loss
            best_checkpoint_epoch = total_epoch
            stale_epochs = 0
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
        else:
            stale_epochs += 1
            if args.patience and stale_epochs >= args.patience:
                print(f"early stopping after {stale_epochs} non-improving validation epochs",
                      flush=True)
                break
    if starting_epoch + args.epochs > 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        project_config = {
            "base_model": base_model_name,
            "context_len": context_len,
            "horizon_len": args.horizon_len,
            "learning_rate": args.lr,
            "epochs_completed": completed_epochs,
            "requested_epochs": starting_epoch + args.epochs,
            "checkpoint_epoch": best_checkpoint_epoch,
            "best_validation_loss": best_val,
            "lora_rank": args.lora_r,
            "seed": args.seed,
            "early_stopping_patience": args.patience,
        }
        (output_dir / "project_config.json").write_text(
            json.dumps(project_config, indent=2) + "\n"
        )
    print(f"Best validation loss: {best_val:.6f}; checkpoint epoch={best_checkpoint_epoch}; "
          f"adapter saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
