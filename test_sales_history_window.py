import json,pytest,pandas as pd
from forecast import main,select_history_window
def test_cli_recent_window_excludes_old_regime_and_discloses_coverage(tmp_path,capsys):
    p=tmp_path/'sales.csv';p.write_text('month,sales\n1,1000\n2,20\n3,30\n4,40\n5,50\n')
    assert main([str(p),'--lookback-periods','3','--json'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['forecast'][0]['predicted_sales']==pytest.approx(60)
    assert data['summary']['training_periods']==3 and data['summary']['available_history_periods']==5
def test_invalid_window_is_rejected():
    with pytest.raises(ValueError):select_history_window(pd.DataFrame({'month':[1,2],'sales':[10,20]}),1)
