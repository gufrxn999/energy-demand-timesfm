import unittest

import numpy as np

from src.models import seasonal_naive


class ModelTests(unittest.TestCase):
    def test_seasonal_naive_repeats_last_week_for_longer_horizon(self):
        context = np.arange(24 * 7, dtype=float)

        forecast = seasonal_naive(context, horizon=24 * 8)

        np.testing.assert_array_equal(forecast[:24 * 7], context[-24 * 7:])
        np.testing.assert_array_equal(forecast[24 * 7:], context[-24 * 7:-24 * 6])

    def test_seasonal_naive_uses_daily_pattern_when_context_is_short(self):
        context = np.arange(30, dtype=float)

        forecast = seasonal_naive(context, horizon=48)

        np.testing.assert_array_equal(forecast[:24], context[-24:])
        np.testing.assert_array_equal(forecast[24:], context[-24:])

    def test_seasonal_naive_rejects_empty_context(self):
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            seasonal_naive([], horizon=24)


if __name__ == "__main__":
    unittest.main()
