import pandas as pd
import pytest
from forecast import predict_next_sales, predict_sales_horizon, backtest_forecast


def test_predictions_are_invariant_to_large_month_offset():
    normal = pd.DataFrame({"month": [1, 2, 3, 4], "sales": [0.1, 0.2, 0.3, 0.4]})
    shifted = normal.copy()
    shifted["month"] += 10**15
    assert predict_next_sales(shifted) == pytest.approx(predict_next_sales(normal), abs=1e-12)
    assert predict_sales_horizon(shifted, 3)["predicted_sales"].tolist() == pytest.approx([0.5, 0.6, 0.7], abs=1e-12)
    assert backtest_forecast(shifted)["backtest_mae"] < 1e-12
