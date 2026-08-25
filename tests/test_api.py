from pathlib import Path
import sys

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import api


client = TestClient(api.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "swot-internal-wave-yolo"
    assert data["model"] == "last.pt"
    assert data["filter_type"] == "rolling"


def test_predict_missing_granule():
    response = client.post(
        "/predict",
        json={
            "granule": "/does/not/exist.nc",
            "confidence": 0.4,
        },
    )

    assert response.status_code == 404


def test_predict_response_contract(
    tmp_path,
    monkeypatch,
):
    # The API checks that the source file exists.
    fake_granule = tmp_path / "test.nc"
    fake_granule.touch()

    def fake_run_johnny_on_one_record(
        detector,
        record,
        output_dir,
        confidence_threshold,
    ):
        return {
            "item_id": record["item_id"],
            "cycle": record["cycle"],
            "pass": record["pass"],
            "mask_netcdf": str(
                output_dir / "fake_mask.nc"
            ),
            "detection_count": 1,
            "filter_type": "rolling",
        }

    def fake_mask_netcdf_to_features(record):
        return [
            {
                "type": "Feature",
                "properties": {
                    "model_id": "johnny_iw",
                    "confidence_value": 0.91,
                    "mask_netcdf": record[
                        "mask_netcdf"
                    ],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [90.0, 10.0],
                            [91.0, 10.0],
                            [91.0, 11.0],
                            [90.0, 11.0],
                            [90.0, 10.0],
                        ]
                    ],
                },
            }
        ]

    monkeypatch.setattr(
        api,
        "run_johnny_on_one_record",
        fake_run_johnny_on_one_record,
    )

    monkeypatch.setattr(
        api,
        "mask_netcdf_to_features",
        fake_mask_netcdf_to_features,
    )

    response = client.post(
        "/predict",
        json={
            "granule": str(fake_granule),
            "confidence": 0.4,
            "source": {
                "collection":
                    "bay_of_bengal_swot_ssha",

                "item_id":
                    "bay_of_bengal_swot_cycle_050_pass_230",

                "cycle": "050",
                "pass": "230",
                "direction": "descending",
                "datetime": "2024-01-01T00:00:00Z",
            },
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_id"] == "johnny_iw"
    assert data["model"] == "last.pt"

    # Johnny's preprocessing stays fixed.
    assert data["filter_type"] == "rolling"

    assert data["confidence_threshold"] == 0.4

    assert data["detection_count"] == 1

    assert isinstance(
        data["processing_seconds"],
        (int, float),
    )

    assert data["source"]["cycle"] == "050"
    assert data["source"]["pass"] == "230"

    geojson = data["geojson"]

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1

    properties = (
        geojson["features"][0]["properties"]
    )

    assert properties[
        "source_collection"
    ] == "bay_of_bengal_swot_ssha"

    assert properties[
        "source_item_id"
    ] == (
        "bay_of_bengal_swot_cycle_050_pass_230"
    )

    assert properties["cycle"] == "050"
    assert properties["pass"] == "230"
    assert properties["direction"] == "descending"

    # Temporary mask paths must NOT escape the API.
    assert "mask_netcdf" not in properties