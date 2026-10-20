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


if __name__ == "__main__":
    unittest.main()
