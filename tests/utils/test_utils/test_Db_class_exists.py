import pytest
from plf import utils

def test_Db_class_exists():
    assert hasattr(utils, "Db")
