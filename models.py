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

    def __init__(self, context_length: int = 512, horizon: int = 24, adapter: str | None = None):
        try:
            import torch
            from transformers import TimesFm2_5ModelForPrediction
        except ImportError as exc:
            raise RuntimeError(
                "Install TimesFM 2.5 inference dependencies: pip install 'timesfm[torch]' transformers"
            ) from exc
        self.horizon = horizon
        self.context_length = context_length
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.device = device
        self.model = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers", torch_dtype=dtype, device_map=device
        )
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Install PEFT to load an adapter: pip install peft") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def predict(self, context) -> np.ndarray:
        import torch
        values = np.asarray(context, dtype=np.float32)[-self.context_length:]
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.model(past_values=tensor)
        return output.mean_predictions[0, :self.horizon].float().cpu().numpy().astype(float)
