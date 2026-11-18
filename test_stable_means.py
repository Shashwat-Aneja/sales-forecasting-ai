import math
import pytest
from forecast import _stable_mean


def test_finite_forecast_means_do_not_overflow_during_summation():
    assert _stable_mean([1e308] * 100) == pytest.approx(1e308)
    assert _stable_mean([1e308, -1e308]) == 0
    assert _stable_mean([3, 5]) == 4


def test_nonfinite_diagnostics_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        _stable_mean([math.inf])
