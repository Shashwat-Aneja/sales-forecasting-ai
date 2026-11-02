import unittest

import pandas as pd

from forecast import (
    backtest_forecast,
    load_sales_history,
    predict_next_sales,
    predict_sales_horizon,
)


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

    def test_rejects_fractional_month_indexes_in_horizon_forecasts(self):
        history = pd.DataFrame({"month": [1.0, 2.5, 3.0], "sales": [10, 25, 30]})
        with self.assertRaisesRegex(ValueError, "whole-number period indexes"):
            predict_sales_horizon(history, periods=2)

    def test_rejects_fractional_month_indexes_in_backtesting(self):
        history = pd.DataFrame({"month": [1.0, 2.5, 3.0], "sales": [10, 25, 30]})
        with self.assertRaisesRegex(ValueError, "whole-number period indexes"):
            backtest_forecast(history)

    def test_rejects_fractional_month_indexes_in_single_period_forecast(self):
        history = pd.DataFrame({"month": [1.5, 2.5], "sales": [15, 25]})
        with self.assertRaisesRegex(ValueError, "whole-number period indexes"):
            predict_next_sales(history)

    def test_csv_loader_rejects_fractional_month_indexes(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "sales.csv"
            pd.DataFrame(
                {"month": [1, 2.5, 3], "sales": [10, 25, 30]}
            ).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "whole-number period indexes"):
                load_sales_history(csv_path)


if __name__ == "__main__":
    unittest.main()
