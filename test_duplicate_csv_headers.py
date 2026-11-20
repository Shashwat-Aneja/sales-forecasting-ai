from forecast import _reject_duplicate_headers
import pytest
def test_duplicate_headers_are_rejected_instead_of_renamed(tmp_path):
    p=tmp_path/'data.csv';p.write_text('value,value\n1,2\n')
    with pytest.raises(ValueError,match='unique'):_reject_duplicate_headers(p)
def test_bom_header_and_distinct_names_are_supported(tmp_path):
    p=tmp_path/'data.csv';p.write_text('\ufeffvalue,other\n1,2\n',encoding='utf-8')
    _reject_duplicate_headers(p)

def test_duplicate_sales_columns_do_not_select_wrong_history(tmp_path):
    from forecast import load_sales_history
    p=tmp_path/'data.csv';p.write_text('month,sales,sales\n1,10,999\n2,20,999\n')
    with pytest.raises(ValueError,match='unique'):load_sales_history(p)
