from pathlib import Path
import sys

from fastapi.testclient import TestClient


# Add the repository root to Python's import path so tests can find api.py
# whether they are run with `pytest` or `python -m pytest`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import app


client = TestClient(app)

TEST_GRANULE = (
    ROOT
    / "test-data"
    / "SWOT_L3_LR_SSH_Expert_001_161_20230726T224518_20230726T233644_v3.0.nc"
)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "swot-internal-wave-yolo"
    assert data["model"] == "last.pt"


def test_predict_missing_granule():
    response = client.post(
        "/predict",
        json={
            "granule": "/does/not/exist.nc",
            "confidence": 0.25,
            "filter_type": "normal",
        },
    )

    assert response.status_code == 404


def test_predict_invalid_filter():
    response = client.post(
        "/predict",
        json={
            "granule": str(TEST_GRANULE),
            "confidence": 0.25,
            "filter_type": "not-a-filter",
        },
    )

    assert response.status_code == 400


def test_predict():
    response = client.post(
        "/predict",
        json={
            "granule": str(TEST_GRANULE),
            "confidence": 0.25,
            "filter_type": "normal",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model"] == "last.pt"
    assert data["filter_type"] == "normal"
    assert data["confidence_threshold"] == 0.25

    assert isinstance(data["detection_count"], int)
    assert isinstance(data["processing_seconds"], (int, float))
    assert isinstance(data["detections"], list)

    assert "summary" in data