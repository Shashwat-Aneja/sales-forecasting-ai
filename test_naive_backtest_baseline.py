import pandas as pd,pytest
from forecast import backtest_forecast
def test_linear_trend_beats_last_observation_on_same_heldout_folds():
    result=backtest_forecast(pd.DataFrame({'month':[1,2,3,4],'sales':[10,20,30,40]}))
    assert result['backtest_naive_mae']==pytest.approx(10)
    assert result['backtest_naive_rmse']==pytest.approx(10)
    assert result['backtest_mae_skill_percent']==pytest.approx(100)
def test_zero_error_baseline_has_undefined_skill_instead_of_division_by_zero():
    result=backtest_forecast(pd.DataFrame({'month':[1,2,3,4],'sales':[10,10,10,10]}))
    assert result['backtest_naive_mae']==0 and result['backtest_mae_skill_percent'] is None
