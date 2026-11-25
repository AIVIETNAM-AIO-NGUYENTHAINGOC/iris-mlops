"""API tests via TestClient with the local model. Run src/model/train.py first."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "local")
    with TestClient(app) as c:  # `with` runs lifespan, which loads the model
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    res = client.post("/predict", json=SETOSA)
    assert res.status_code == 200
    assert res.json()["class"] == "setosa"


def test_predict_rejects_invalid_input(client):
    res = client.post("/predict", json={**SETOSA, "petal_width": -1})
    assert res.status_code == 422
