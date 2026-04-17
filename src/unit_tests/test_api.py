import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from api import app

TEST_IMAGE = os.path.join("data", "train", "cat.0.jpg")


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_returns_200(client):
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/predict", data={"file": f})
    assert response.status_code == 200


def test_predict_response_structure(client):
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/predict", data={"file": f})
    data = response.get_json()
    assert "label" in data
    assert "confidence" in data


def test_predict_label_valid(client):
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/predict", data={"file": f})
    label = response.get_json()["label"]
    assert label in {"cat", "dog"}


def test_predict_no_file(client):
    response = client.post("/predict")
    assert response.status_code == 400
