import pytest
from plf import danger

def test_corrupt_ppl_exists():
    assert hasattr(danger, "corrupt_ppl")
