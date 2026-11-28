import pytest
from forecast import _stable_mean

def test_cancellation_and_overflow_means():
    assert _stable_mean([1e16, -9999999999999998.0]) == 1
    assert _stable_mean([1e308, 1e-100, -1e308]) == pytest.approx(1e-100/3, rel=1e-12, abs=0)
    assert _stable_mean([1e308, 1e308]) == 1e308
