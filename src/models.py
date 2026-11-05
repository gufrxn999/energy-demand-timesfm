"""Forecast model adapters."""
import numpy as np


def seasonal_naive(context, horizon: int, season: int = 24 * 7) -> np.ndarray:
    """Repeat the last observed weekly pattern, wrapping for longer horizons."""
    values = np.asarray(context, dtype=float)
    if len(values) < season:
        season = min(24, len(values))
    if season < 1:
        raise ValueError("Context must contain at least one observation")
    return np.resize(values[-season:], horizon)


class TimesFMZeroShot:
    """TimesFM 2.5 Transformers inference, optionally with a trained PEFT adapter."""

    def __init__(self, context_length: int = 512, horizon: int = 24,
                 adapter: str | None = None,
                 model_id: str = "google/timesfm-2.5-200m-transformers"):
        try:
            import torch
            from transformers import TimesFm2_5ModelForPrediction
        except ImportError as exc:
            raise RuntimeError(
                "Install TimesFM 2.5 inference dependencies: pip install 'timesfm[torch]' transformers"
            ) from exc
        self.horizon = horizon
        self.context_length = context_length
        self.model_id = model_id
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.device = device
        self.model = TimesFm2_5ModelForPrediction.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device
        )
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Install PEFT to load an adapter: pip install peft") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def predict(self, context) -> np.ndarray:
        return self.predict_many([context], batch_size=1)[0]

    def predict_many(self, contexts, batch_size: int = 8) -> np.ndarray:
        import torch
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        arrays = [np.asarray(context, dtype=np.float32)[-self.context_length:]
                  for context in contexts]
        if not arrays:
            return np.empty((0, self.horizon), dtype=float)
        if len({len(values) for values in arrays}) != 1:
            return np.stack([self.predict(values) for values in arrays])
        forecasts = []
        with torch.no_grad():
            for start in range(0, len(arrays), batch_size):
                batch = np.stack(arrays[start:start + batch_size])
                tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
                output = self.model(past_values=tensor)
                forecasts.append(output.mean_predictions[:, :self.horizon].float().cpu().numpy())
        return np.concatenate(forecasts).astype(float)
