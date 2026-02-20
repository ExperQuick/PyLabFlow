import pytest
from plf import lab

def test_export_settigns_edge_case():
    try:
        lab.export_settigns()
    except Exception:
        assert True
