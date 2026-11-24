import pandas as pd
from forecast import summarize_forecast

def test_summary_discloses_unobserved_months_without_imputing_sales():
    history=pd.DataFrame({'month':[5,1,3],'sales':[50,10,30]})
    result=summarize_forecast(history,60)
    assert result['history_period_span']==5 and result['missing_history_periods']==2
    assert result['historical_average']==30
