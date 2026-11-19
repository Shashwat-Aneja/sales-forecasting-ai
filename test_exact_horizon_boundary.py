import pandas as pd
import pytest
from forecast import MAX_SAFE_MONTH_INDEX as M, predict_sales_horizon, summarize_forecast
def test_last_exact_month_is_available_once():
    df=pd.DataFrame({'month':[M-2,M-1],'sales':[10,20]})
    result=predict_sales_horizon(df,1)
    assert result['month'].tolist()==[float(M)]
    with pytest.raises(ValueError):predict_sales_horizon(df,2)
def test_summary_rejects_unrepresentable_next_month():
    with pytest.raises(ValueError):summarize_forecast(pd.DataFrame({'month':[M-1,M],'sales':[10,20]}),30)
