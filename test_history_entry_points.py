import pandas as pd
import pytest
from forecast import summarize_forecast, predict_next_sales, predict_sales_horizon, backtest_forecast


def test_summary_handles_non_unique_dataframe_indexes():
    history = pd.DataFrame({"month": [3, 1, 2], "sales": [30, 10, 20]}, index=[0, 0, 0])
    assert summarize_forecast(history, 40)["trend"] == "increasing"


@pytest.mark.parametrize("function", [predict_next_sales, predict_sales_horizon, backtest_forecast])
def test_public_forecasters_reject_boolean_history(function):
    with pytest.raises(ValueError, match="booleans"):
        function(pd.DataFrame({"month": [True, False], "sales": [10, 20]}))
