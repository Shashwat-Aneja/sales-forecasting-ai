import pandas as pd
import pytest
from forecast import predict_sales_horizon, MAX_FORECAST_PERIODS


def test_horizon_limit_is_checked_before_allocating_forecast():
    history = pd.DataFrame({"month": [1, 2], "sales": [1, 2]})
    with pytest.raises(ValueError, match="cannot exceed"):
        predict_sales_horizon(history, 10**12)
    assert len(predict_sales_horizon(history, MAX_FORECAST_PERIODS)) == MAX_FORECAST_PERIODS
