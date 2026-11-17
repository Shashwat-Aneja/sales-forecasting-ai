import numpy as np
import pandas as pd
import pytest
import forecast


@pytest.mark.parametrize("entry", [forecast.predict_next_sales, forecast.predict_sales_horizon, forecast.backtest_forecast])
def test_nonfinite_model_output_is_rejected(entry, monkeypatch):
    monkeypatch.setattr(forecast.LinearRegression, "predict", lambda self, x: np.full(len(x), np.inf))
    history = pd.DataFrame({"month": [1, 2, 3], "sales": [10, 20, 30]})
    with pytest.raises(ValueError, match="numeric range"):
        entry(history)
