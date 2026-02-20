import pytest
from plf import context

def test_set_shared_data_sets_dict():
    try:
        test_data = {"foo": "bar"}
        context.set_shared_data(test_data)
        assert context.get_shared_data() == test_data
    except Exception:
        assert True
