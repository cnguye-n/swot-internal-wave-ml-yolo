from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from swot_internal_wave_detector import SWOTInternalWaveDetector
from pipeline.step_02_run_johnny_iw import (
    JOHNNY_FILTER_TYPE,
    run_johnny_on_one_record,
)
from pipeline.step_03_yolo_to_geojson import (
    mask_netcdf_to_features,
)


TEST_GRANULE = (
    ROOT
    / "test-data"
    / "SWOT_L3_LR_SSH_Expert_001_161_20230726T224518_20230726T233644_v3.0.nc"
)

MODEL_PATH = ROOT / "last.pt"


def make_test_record():
    return {
        "item_id": "test_cycle_001_pass_161",
        "collection": "bay_of_bengal_swot_ssha",
        "cycle": "001",
        "pass": "161",
        "direction": "ascending",
        "datetime": None,
        "netcdf_path": str(TEST_GRANULE),
    }


@pytest.fixture(scope="module")
def model_result(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp(
        "johnny_model_output"
    )

    detector = SWOTInternalWaveDetector(
        lat_min=3.0,
        lat_max=15.5,
        model_path=MODEL_PATH,
    )

    result = run_johnny_on_one_record(
        detector=detector,
        record=make_test_record(),
        output_dir=output_dir,
        confidence_threshold=0.4,
    )

    return result


def test_johnny_model_runs(model_result):
    assert model_result["cycle"] == "001"
    assert model_result["pass"] == "161"

    assert (
        model_result["filter_type"]
        == JOHNNY_FILTER_TYPE
    )

    assert JOHNNY_FILTER_TYPE == "rolling"

    assert isinstance(
        model_result["detection_count"],
        int,
    )

    assert Path(
        model_result["mask_netcdf"]
    ).exists()

    assert Path(
        model_result["summary_text"]
    ).exists()

    assert Path(
        model_result["boxed_preview"]
    ).exists()


def test_model_output_becomes_geojson(
    model_result,
):
    features = mask_netcdf_to_features(
        model_result
    )

    assert isinstance(features, list)

    for feature in features:
        assert feature["type"] == "Feature"

        assert (
            feature["geometry"]["type"]
            == "Polygon"
        )

        assert "properties" in feature