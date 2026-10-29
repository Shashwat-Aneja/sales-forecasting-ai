import pandas as pd
import pytest

from forecast import backtest_forecast


def test_backtest_uses_held_out_periods_and_reports_error_metrics():
    history = pd.DataFrame(
        {"month": [1, 2, 3, 4], "sales": [10, 20, 30, 50]}
    )

    result = backtest_forecast(history)

    assert result["backtest_periods"] == 2
    assert abs(result["backtest_mae"] - 5.0) < 1e-9
    assert abs(result["backtest_rmse"] - (50 ** 0.5)) < 1e-9


def test_backtest_is_unavailable_without_three_observations():
    history = pd.DataFrame({"month": [1, 2], "sales": [10, 20]})

    assert backtest_forecast(history) is None


def test_backtest_sorts_observations_before_splitting():
    history = pd.DataFrame(
        {"month": [4, 2, 3, 1], "sales": [50, 20, 30, 10]}
    )

    result = backtest_forecast(history)

    assert result["backtest_periods"] == 2
    assert abs(result["backtest_mae"] - 5.0) < 1e-9
