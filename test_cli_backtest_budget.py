import json,pytest
from forecast import main

def test_cli_bounds_rolling_fit_work_and_reports_coverage(tmp_path,capsys):
    p=tmp_path/'sales.csv';p.write_text('month,sales\n1,10\n2,20\n3,30\n4,40\n5,50\n')
    assert main([str(p),'--backtest-periods','1','--json'])==0
    summary=json.loads(capsys.readouterr().out)['summary']
    assert summary['backtest_periods']==1 and summary['backtest_available_periods']==3
    assert summary['backtest_start_month']==5

def test_cli_rejects_zero_backtest_budget_before_reading():
    with pytest.raises(SystemExit):main(['missing.csv','--backtest-periods','0'])
