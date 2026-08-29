from pathlib import Path
import sys

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api


client = TestClient(api.app)

TEST_FILENAME = (
    "SWOT_L3_LR_SSH_Expert_001_161_"
    "20230726T224518_20230726T233644_v3.0.nc"
)

TEST_DATA_ROOT = ROOT / "test-data"


def test_real_prediction(monkeypatch):
    test_granule = TEST_DATA_ROOT / TEST_FILENAME

    assert test_granule.is_file(), f"Missing test granule: {test_granule}"

    # The API normally sees a Docker mount such as /data.
    # For this local integration test, point it at repo/test-data.
    monkeypatch.setattr(api, "DATA_ROOT", TEST_DATA_ROOT)

    response = client.post(
        "/predict",
        json={
            "input": {
                "type": "stac_item",
                "collection": "bay_of_bengal_swot_ssha",
                "item_id": "test_cycle_001_pass_161",
                "source_file": TEST_FILENAME,
            },
            "parameters": {
                "confidenceThreshold": 0.4,
            },
            "source": {
                "collection": "bay_of_bengal_swot_ssha",
                "item_id": "test_cycle_001_pass_161",
                "cycle": "001",
                "pass": "161",
                "direction": "ascending",
                "variable": "ssha_unfiltered",
            },
        },
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "swot-internal-wave-yolo"
    assert data["model_id"] == "iw_yolo"
    assert data["filter_type"] == "rolling"
    assert data["confidence_threshold"] == 0.4
    assert isinstance(data["processing_seconds"], (int, float))

    geojson = data["geojson"]

    assert geojson["type"] == "FeatureCollection"
    assert isinstance(geojson["features"], list)
    assert data["detection_count"] == len(geojson["features"])

    for feature in geojson["features"]:
        properties = feature["properties"]

        assert properties["model_id"] == "iw_yolo"
        assert properties["id"].startswith("iw_yolo_cycle_")
        assert "confidence_value" in properties
        assert "processing_seconds" in properties
        assert "mask_netcdf" not in properties