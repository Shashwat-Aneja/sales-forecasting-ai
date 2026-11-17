import numpy as np
import pandas as pd
import pytest
import forecast


def test_large_finite_residuals_keep_finite_rmse():
    assert forecast._stable_rmse([1e200, -1e200]) == pytest.approx(1e200)
    assert forecast._stable_rmse([0, 0]) == 0
    assert forecast._stable_rmse([3, 4]) == pytest.approx(np.sqrt(12.5))


def test_backtest_metrics_do_not_overflow_when_squared(monkeypatch):
    history = pd.DataFrame({"month": [1, 2, 3, 4], "sales": [1e200] * 4})
    monkeypatch.setattr(forecast.LinearRegression, "predict", lambda self, values: np.zeros(len(values)))
    assert forecast.backtest_forecast(history)["backtest_rmse"] == pytest.approx(1e200)
    assert forecast.summarize_forecast(history, 1e200)["historical_fit_rmse"] == pytest.approx(1e200)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_residuals_are_validation_errors(value):
    with pytest.raises(ValueError, match="residuals"):
        forecast._stable_rmse([value])
