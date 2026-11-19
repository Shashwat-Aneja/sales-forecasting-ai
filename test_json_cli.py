import json
from forecast import main
def test_json_forecast_includes_horizon_and_backtest(tmp_path,capsys):
    p=tmp_path/'sales.csv';p.write_text('month,sales\n1,10\n2,20\n3,30\n')
    assert main([str(p),'--horizon','2','--json'])==0
    data=json.loads(capsys.readouterr().out)
    assert len(data['forecast'])==2
    assert data['summary']['backtest_periods']==1
