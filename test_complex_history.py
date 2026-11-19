import pandas as pd
import pytest
from forecast import validate_history
@pytest.mark.parametrize('column',['month','sales'])
def test_complex_history_does_not_discard_imaginary_component(column):
    df=pd.DataFrame({'month':[1,2],'sales':[10,20]});df[column]=[1+2j,2+3j]
    with pytest.raises(ValueError,match='real numbers'):validate_history(df)
