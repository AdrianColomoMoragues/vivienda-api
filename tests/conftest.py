import importlib
import os

import pytest
from fastapi.testclient import TestClient

from vivienda.train import make_fake, train


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    train(make_fake(500), out, cv_folds=3, importancia=False)
    return out


@pytest.fixture(scope="session")
def client(models_dir):
    os.environ["MODELS_DIR"] = str(models_dir)
    import vivienda.api as api
    importlib.reload(api)                  # vuelve a leer MODELS_DIR
    with TestClient(api.app) as c:
        yield c
