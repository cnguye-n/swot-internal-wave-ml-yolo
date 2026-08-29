from pathlib import Path
import sys

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api


client = TestClient(api.app)


def make_request(source_file="test.nc"):
    return {
        "input": {
            "type": "stac_item",
            "collection": "bay_of_bengal_swot_ssha",
            "item_id": "bay_of_bengal_swot_cycle_050_pass_230",
            "source_file": source_file,
        },
        "parameters": {
            "confidenceThreshold": 0.4,
        },
        "source": {
            "collection": "bay_of_bengal_swot_ssha",
            "item_id": "bay_of_bengal_swot_cycle_050_pass_230",
            "cycle": "050",
            "pass": "230",
            "direction": "descending",
            "datetime": "2024-01-01T00:00:00Z",
            "variable": "ssha_unfiltered",
        },
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "swot-internal-wave-yolo"
    assert data["model"] == "last.pt"
    assert data["filter_type"] == "rolling"


def test_metadata():
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert "summary" in data
    assert "feature_properties" in data

    summary_fields = {row["field"] for row in data["summary"]}
    feature_fields = {row["field"] for row in data["feature_properties"]}

    assert "detection_count" in summary_fields
    assert "id" in feature_fields
    assert "confidence_value" in feature_fields
    assert "processing_seconds" in feature_fields


def test_predict_missing_granule(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_ROOT", tmp_path)

    response = client.post("/predict", json=make_request("missing.nc"))

    assert response.status_code == 404
    assert "Granule not found" in response.json()["detail"]


def test_predict_response_contract(tmp_path, monkeypatch):
    cycle_dir = tmp_path / "cycle_050"
    cycle_dir.mkdir()

    fake_granule = cycle_dir / "test.nc"
    fake_granule.touch()

    monkeypatch.setattr(api, "DATA_ROOT", tmp_path)

    def fake_run_iw_yolo_on_one_record(detector, record, output_dir, confidence_threshold):
        return {
            "item_id": record["item_id"],
            "cycle": record["cycle"],
            "pass": record["pass"],
            "mask_netcdf": str(output_dir / "fake_mask.nc"),
            "detection_count": 1,
            "filter_type": "rolling",
        }

    def fake_mask_netcdf_to_features(record):
        return [
            {
                "type": "Feature",
                "properties": {
                    "id": "iw_yolo_cycle_050_pass_230_box_001",
                    "model_id": "iw_yolo",
                    "confidence_value": 0.91,
                    "mask_netcdf": record["mask_netcdf"],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [90.0, 10.0],
                        [91.0, 10.0],
                        [91.0, 11.0],
                        [90.0, 11.0],
                        [90.0, 10.0],
                    ]],
                },
            }
        ]

    monkeypatch.setattr(api, "run_iw_yolo_on_one_record", fake_run_iw_yolo_on_one_record)
    monkeypatch.setattr(api, "mask_netcdf_to_features", fake_mask_netcdf_to_features)

    response = client.post("/predict", json=make_request())

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "swot-internal-wave-yolo"
    assert data["model_id"] == "iw_yolo"
    assert data["model"] == "last.pt"
    assert data["filter_type"] == "rolling"
    assert data["confidence_threshold"] == 0.4
    assert data["detection_count"] == 1
    assert isinstance(data["processing_seconds"], (int, float))

    assert data["source"]["cycle"] == "050"
    assert data["source"]["pass"] == "230"
    assert data["source"]["variable"] == "ssha_unfiltered"

    geojson = data["geojson"]

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1

    properties = geojson["features"][0]["properties"]

    assert properties["id"] == "iw_yolo_cycle_050_pass_230_box_001"
    assert properties["source_collection"] == "bay_of_bengal_swot_ssha"
    assert properties["source_item_id"] == "bay_of_bengal_swot_cycle_050_pass_230"
    assert properties["cycle"] == "050"
    assert properties["pass"] == "230"
    assert properties["direction"] == "descending"
    assert isinstance(properties["processing_seconds"], (int, float))

    # Temporary files must never leak through the HTTP response.
    assert "mask_netcdf" not in properties
