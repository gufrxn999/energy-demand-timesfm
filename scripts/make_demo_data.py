"""Create a deterministic synthetic hourly load series for pipeline checks."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(17)
    index = pd.date_range("2024-01-01", periods=24 * 180, freq="h")
    hour = index.hour.to_numpy()
    day = index.dayofweek.to_numpy()
    annual = np.sin(2 * np.pi * np.arange(len(index)) / (24 * 365.25))
    load = (900 + 120 * np.sin(2 * np.pi * (hour - 8) / 24)
            + 55 * (day < 5) + 70 * annual + rng.normal(0, 18, len(index)))
    out = Path("data")
    out.mkdir(exist_ok=True)
    pd.DataFrame({"timestamp": index, "load": load}).to_csv(out / "demo_hourly.csv", index=False)
    print(f"Wrote {out / 'demo_hourly.csv'} ({len(index)} hourly points)")


if __name__ == "__main__":
    main()
