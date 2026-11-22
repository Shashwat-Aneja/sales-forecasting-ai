import pandas as pd,pytest
from forecast import validate_history
def test_object_complex_sales_are_rejected_after_numeric_conversion():
    frame=pd.DataFrame({'month':[1,2],'sales':pd.Series([1+2j,2+3j],dtype=object)})
    with pytest.raises(ValueError,match='real numbers'):validate_history(frame)
