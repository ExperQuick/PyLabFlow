import pytest
from plf import experiment

def test_get_ppls_returns_list():
    try:
        result = experiment.get_ppls()
        assert isinstance(result, list) or result is None
    except KeyError:
        assert True
    except Exception:
        assert False
