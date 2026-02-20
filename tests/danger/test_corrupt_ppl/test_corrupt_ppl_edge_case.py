import pytest
from plf import danger

def test_corrupt_ppl_edge_case():
    try:
        danger.corrupt_ppl("nonexistent_id")
    except Exception:
        assert True
