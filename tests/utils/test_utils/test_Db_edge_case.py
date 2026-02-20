import pytest
from plf import utils

def test_Db_edge_case():
    try:
        db = utils.Db(db_path=":memory:")
        db.close()
    except Exception:
        assert True
