import json
import pandas as pd
import pytest
from forecast import predict_sales_horizon, backtest_forecast, main

def test_forecast_and_backtest_share_policy(tmp_path,capsys):
    history=pd.DataFrame({"month":[1,2,3],"sales":[30,10,0]})
    assert predict_sales_horizon(history).predicted_sales.iloc[0]<0
    assert predict_sales_horizon(history,non_negative=True).predicted_sales.iloc[0]==0
    assert backtest_forecast(history)["backtest_mae"]==pytest.approx(10)
    assert backtest_forecast(history,non_negative=True)["backtest_mae"]==0
    src=tmp_path/"a.csv";history.to_csv(src,index=False)
    assert main([str(src),'--json','--non-negative'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['summary']['predicted_sales']==0 and result['summary']['forecast_policy']=='non_negative' and result['forecast'][0]['predicted_sales']==0
