"""LoRA fine-tuning for TimesFM 2.5 on a single hourly load CSV.

Training windows come only from the training partition. The final chronological
15% is held out for evaluation and is never used for adapter selection.
"""
import argparse
from pathlib import Path
import sys

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
    args = parser.parse_args()
    if args.context_len % 32:
        raise ValueError("TimesFM 2.5 context length must be a multiple of 32")

    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    split = chronological_split(load_hourly_csv(args.data))
    context_len = args.context_len
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map=device
    )
    context_len = min(context_len, model.config.context_length)
    if context_len % 32:
        context_len = context_len - context_len % 32
    train_ds = WindowDataset(split.train, context_len, args.horizon_len, args.samples, args.seed)
    val_ds = WindowDataset(split.validation, context_len, args.horizon_len,
                           min(256, max(1, len(split.validation) // (context_len + args.horizon_len))),
                           args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear",
        lora_dropout=0.05, bias="none",
    ))
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output_dir = Path(args.output_dir)
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            loss = model(past_values=past, future_values=future,
                         forecast_context_len=context_len).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        model.eval()
        losses = []
        with torch.no_grad():
            for past, future in val_loader:
                out = model(past_values=past.to(device), future_values=future.to(device),
                            forecast_context_len=context_len)
                losses.append(float(out.loss))
        val_loss = float(np.mean(losses))
        print(f"epoch={epoch + 1}/{args.epochs} validation_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            (output_dir / "project_config.json").write_text(
                '{\n'
                f'  "base_model": "{args.model_id}",\n'
                f'  "context_len": {context_len},\n'
                f'  "horizon_len": {args.horizon_len},\n'
                f'  "learning_rate": {args.lr},\n'
                f'  "epochs_completed": {epoch + 1},\n'
                f'  "lora_rank": {args.lora_r},\n'
                f'  "seed": {args.seed}\n'
                '}\n'
            )
    print(f"Best validation loss: {best_val:.6f}; adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
