import tempfile
import unittest
from pathlib import Path

import pandas as pd

from forecast import load_sales_history, predict_next_sales


class SalesForecastTests(unittest.TestCase):
    def write_csv(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return handle.name

    def test_history_is_sorted_by_month(self):
        path = self.write_csv("month,sales\n3,30\n1,10\n2,20\n")
        result = load_sales_history(path)
        self.assertEqual(result["month"].tolist(), [1, 2, 3])

    def test_duplicate_months_are_rejected(self):
        path = self.write_csv("month,sales\n1,10\n1,12\n2,20\n")
        with self.assertRaisesRegex(ValueError, "only once"):
            load_sales_history(path)

    def test_non_numeric_sales_are_rejected(self):
        path = self.write_csv("month,sales\n1,10\n2,unknown\n")
        with self.assertRaisesRegex(ValueError, "numeric"):
            load_sales_history(path)

    def test_linear_trend_predicts_next_month(self):
        history = pd.DataFrame({"month": [1, 2, 3], "sales": [10, 20, 30]})
        self.assertAlmostEqual(predict_next_sales(history), 40.0)


    def test_forecast_summary_reports_average_period_and_trend(self):
        from forecast import summarize_forecast

        history = pd.DataFrame({"month": [1, 2, 3], "sales": [10, 20, 30]})
        summary = summarize_forecast(history, 40.0)

        self.assertEqual(summary["next_month"], 4.0)
        self.assertEqual(summary["historical_average"], 20.0)
        self.assertEqual(summary["predicted_sales"], 40.0)
        self.assertEqual(summary["trend"], "increasing")

    def test_forecast_sales_prints_context_metrics(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from forecast import forecast_sales

        path = self.write_csv("month,sales\n1,10\n2,20\n3,30\n")
        output = StringIO()
        with redirect_stdout(output):
            forecast_sales(path)
        self.assertIn("Historical average: ₹20.00", output.getvalue())
        self.assertIn("Trend: increasing", output.getvalue())

    def test_forecast_summary_reports_in_sample_fit_errors(self):
        from forecast import summarize_forecast

        history = pd.DataFrame({"month": [1, 2, 3], "sales": [10, 20, 40]})
        summary = summarize_forecast(history, 50.0)

        self.assertGreater(summary["historical_fit_mae"], 0)
        self.assertGreater(summary["historical_fit_rmse"], 0)
        self.assertGreaterEqual(
            summary["historical_fit_rmse"], summary["historical_fit_mae"]
        )

    def test_forecast_summary_marks_flat_trend(self):
        from forecast import summarize_forecast

        history = pd.DataFrame({"month": [1, 2], "sales": [10, 10]})
        self.assertEqual(summarize_forecast(history, 10)["trend"], "stable")

if __name__ == "__main__":
    unittest.main()
