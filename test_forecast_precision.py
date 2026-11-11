import pandas as pd
import pytest

from forecast import load_sales_history, predict_next_sales, predict_sales_horizon


def test_csv_rejects_months_beyond_exact_float_integer_range(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("month,sales\n1,10\n9007199254740992,20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integer precision"):
        load_sales_history(path)


def test_next_month_rejects_precision_boundary():
    history = pd.DataFrame({"month": [9007199254740990, 9007199254740991], "sales": [10, 20]})
    with pytest.raises(ValueError, match="integer precision"):
        predict_next_sales(history)


def test_horizon_rejects_precision_boundary_before_generating_months():
    history = pd.DataFrame({"month": [9007199254740989, 9007199254740990], "sales": [10, 20]})
    with pytest.raises(ValueError, match="integer precision"):
        predict_sales_horizon(history, periods=2)
