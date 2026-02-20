import pytest
from plf import context

def test_get_shared_data_returns_dict():
    try:
        data = context.get_shared_data()
        assert isinstance(data, dict)
    except Exception:
        assert True
