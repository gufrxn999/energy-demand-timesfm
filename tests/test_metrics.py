import unittest

from src.metrics import score


class MetricTests(unittest.TestCase):
    def test_score_reports_mae_rmse_and_mase(self):
        result = score(actual=[1, 2, 3], predicted=[2, 2, 4], seasonal_scale=2)

        self.assertAlmostEqual(result["MAE"], 2 / 3)
        self.assertAlmostEqual(result["RMSE"], (2 / 3) ** 0.5)
        self.assertAlmostEqual(result["MASE"], 1 / 3)

    def test_score_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            score(actual=[1, 2], predicted=[1], seasonal_scale=1)

    def test_mase_is_nan_when_seasonal_scale_is_zero(self):
        result = score(actual=[1], predicted=[1], seasonal_scale=0)

        self.assertTrue(result["MASE"] != result["MASE"])


if __name__ == "__main__":
    unittest.main()
