import pandas as pd
import forecast
def test_backtest_budget_evaluates_latest_folds_without_future_leakage(monkeypatch):
    history=pd.DataFrame({'month':range(1,41),'sales':range(10,410,10)})
    fitted=[];original=forecast._fit_scaled_trend
    def capture(model,months,sales):fitted.append(len(sales));return original(model,months,sales)
    monkeypatch.setattr(forecast,'_fit_scaled_trend',capture)
    result=forecast.backtest_forecast(history,max_periods=3)
    assert fitted==[37,38,39] and result['backtest_periods']==3
    assert result['backtest_available_periods']==38 and result['backtest_start_month']==38
    assert result['backtest_rmse']<1e-9
