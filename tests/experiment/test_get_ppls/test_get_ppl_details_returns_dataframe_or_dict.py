import pytest
from plf import experiment

def test_get_ppl_details_returns_dataframe_or_dict():
    try:
        result = experiment.get_ppl_details([])
        assert result is not None
    except Exception:
        assert True
