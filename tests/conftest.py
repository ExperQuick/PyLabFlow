# tests/conftest.py

import pytest
import os
import json
from plf.lab import create_project, lab_setup

@pytest.fixture(scope="session")
def setup_lab_env(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("lab_project")

    settings = {
        "project_name": "FixtureLab",
        "project_dir": str(tmp_dir),
        "component_dir": str(tmp_dir / "components"),
        "metrics": ["accuracy", "loss"],
        "strategy": {"monitor": "val_loss", "mode": "min"},
    }

    settings_path = create_project(settings)
    lab_setup(settings_path)

    return {
        "settings": settings,
        "path": tmp_dir
    }
