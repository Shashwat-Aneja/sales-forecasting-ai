import pandas as pd
import pytest
from forecast import predict_next_sales,predict_sales_horizon,summarize_forecast
def test_large_constant_sales_fit_and_metrics_remain_finite():
    history=pd.DataFrame({'month':[1,2,3,4],'sales':[1e308]*4})
    predicted=predict_next_sales(history)
    assert predicted==pytest.approx(1e308)
    assert predict_sales_horizon(history,2)['predicted_sales'].tolist()==pytest.approx([1e308,1e308])
    summary=summarize_forecast(history,predicted)
    assert summary['historical_fit_rmse']==0 and summary['backtest_rmse']==0
