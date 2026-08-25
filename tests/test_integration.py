from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import app


client = TestClient(app)


TEST_GRANULE = (
    ROOT
    / "test-data"
    / "SWOT_L3_LR_SSH_Expert_001_161_20230726T224518_20230726T233644_v3.0.nc"
)


@pytest.mark.integration
def test_real_prediction():
    response = client.post(
        "/predict",
        json={
            "granule": str(TEST_GRANULE),
            "confidence": 0.4,
            "source": {
                "collection":
                    "bay_of_bengal_swot_ssha",
                "item_id":
                    "test_cycle_001_pass_161",
                "cycle": "001",
                "pass": "161",
                "direction": "ascending",
            },
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_id"] == "johnny_iw"
    assert data["filter_type"] == "rolling"

    assert isinstance(
        data["processing_seconds"],
        (int, float),
    )

    geojson = data["geojson"]

    assert geojson["type"] == (
        "FeatureCollection"
    )

    assert isinstance(
        geojson["features"],
        list,
    )

    assert data["detection_count"] == len(
        geojson["features"]
    )