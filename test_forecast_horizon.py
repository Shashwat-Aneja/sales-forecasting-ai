import unittest

import pandas as pd

from forecast import predict_sales_horizon


class ForecastHorizonTests(unittest.TestCase):
    def test_predicts_multiple_future_months_in_order(self):
        history = pd.DataFrame({"month": [3, 1, 2], "sales": [30, 10, 20]})
        result = predict_sales_horizon(history, periods=3)
        self.assertEqual(result["month"].tolist(), [4.0, 5.0, 6.0])
        for actual, expected in zip(result["predicted_sales"], [40.0, 50.0, 60.0]):
            self.assertAlmostEqual(actual, expected)

    def test_rejects_non_positive_or_non_integer_horizons(self):
        history = pd.DataFrame({"month": [1, 2], "sales": [10, 20]})
        for periods in (0, -1, True, 1.5):
            with self.subTest(periods=periods):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    predict_sales_horizon(history, periods=periods)

    def test_rejects_duplicate_months(self):
        history = pd.DataFrame({"month": [1, 2, 2], "sales": [10, 20, 30]})
        with self.assertRaisesRegex(ValueError, "only once"):
            predict_sales_horizon(history, periods=2)


if __name__ == "__main__":
    unittest.main()
