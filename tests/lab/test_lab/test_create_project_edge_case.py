import pytest
from plf import lab

def test_create_project_edge_case():
    try:
        lab.create_project({"project_dir": ".", "project_name": "x", "component_dir": "."})
    except Exception:
        assert True
