import pytest
from plf import lab

def test_create_project_exists():
    assert hasattr(lab, "create_project")
