import json,pandas as pd,pytest
from forecast import main,export_forecast_csv
def test_json_cli_also_exports_every_period(tmp_path,capsys):
    source=tmp_path/'history.csv';source.write_text('month,sales\n1,10\n2,20\n3,30\n')
    output=tmp_path/'out.csv'
    assert main([str(source),'--horizon','3','--json','--output',str(output)])==0
    assert len(json.loads(capsys.readouterr().out)['forecast'])==3
    assert len(pd.read_csv(output))==3
def test_failed_export_preserves_previous_report(tmp_path,monkeypatch):
    source=tmp_path/'history.csv';source.write_text('month,sales\n1,10\n2,20\n')
    output=tmp_path/'out.csv';output.write_text('keep')
    def fail(*a,**k):raise OSError('write failed')
    monkeypatch.setattr(pd.DataFrame,'to_csv',fail)
    with pytest.raises(OSError):export_forecast_csv(pd.DataFrame({'value':[1]}),output,source)
    assert output.read_text()=='keep'
def test_source_alias_is_rejected(tmp_path):
    source=tmp_path/'history.csv';source.write_text('keep')
    with pytest.raises(ValueError):export_forecast_csv(pd.DataFrame(),source,source)
    assert source.read_text()=='keep'
