import pytest
from plf import utils

def test_hash_args_exists():
    assert hasattr(utils, "hash_args")
