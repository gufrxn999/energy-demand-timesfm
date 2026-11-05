import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import chronological_split, load_hourly_csv


class DataTests(unittest.TestCase):
    def test_load_hourly_csv_sorts_and_forward_fills_short_gap(self):
        timestamps = pd.date_range("2025-01-01", periods=24 * 16, freq="h")
        frame = pd.DataFrame({"timestamp": timestamps, "load": range(len(timestamps))})
        frame = frame.drop(index=100).sample(frac=1, random_state=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "load.csv"
            frame.to_csv(path, index=False)
            series = load_hourly_csv(path)

        self.assertEqual(len(series), len(timestamps))
        self.assertTrue(series.index.is_monotonic_increasing)
        self.assertAlmostEqual(series.loc[timestamps[100]], 99.0)

    def test_load_hourly_csv_rejects_short_series(self):
        timestamps = pd.date_range("2025-01-01", periods=24 * 14, freq="h")
        frame = pd.DataFrame({"timestamp": timestamps, "load": range(len(timestamps))})

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "short.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "at least 15 days"):
                load_hourly_csv(path)

    def test_chronological_split_preserves_order_and_fractions(self):
        series = pd.Series(range(100), index=pd.date_range("2025-01-01", periods=100, freq="h"))

        split = chronological_split(series)

        self.assertEqual((len(split.train), len(split.validation), len(split.test)), (70, 15, 15))
        self.assertLess(split.train.index[-1], split.validation.index[0])
        self.assertLess(split.validation.index[-1], split.test.index[0])


if __name__ == "__main__":
    unittest.main()
