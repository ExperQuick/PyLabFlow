import pytest
from plf import lab

def test_export_settigns_exists():
    assert hasattr(lab, "export_settigns")
