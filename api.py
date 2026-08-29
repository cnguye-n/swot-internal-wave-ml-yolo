import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from swot_internal_wave_detector import SWOTInternalWaveDetector
from pipeline.step_02_run_iw import IW_YOLO_FILTER_TYPE, run_iw_yolo_on_one_record
from pipeline.step_03_yolo_to_geojson import mask_netcdf_to_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "last.pt"
UI_CONFIG_PATH = BASE_DIR / "model_ui_for_mmgis.json"
DATA_ROOT = Path(os.environ.get("SWOT_DATA_ROOT", "/data"))


app = FastAPI(
    title="SWOT Internal Wave YOLO API",
    version="1.0.0",
)


# Load the model once when the API starts.
detector = SWOTInternalWaveDetector(
    lat_min=3.0,
    lat_max=15.5,
    model_path=MODEL_PATH,
)


class ModelInput(BaseModel):
    type: Literal["stac_item"]
    collection: str
    item_id: str | None = None
    source_file: str


class ModelParameters(BaseModel):
    confidenceThreshold: float = Field(default=0.4, ge=0.0, le=1.0)


class SourceMetadata(BaseModel):
    collection: str | None = None
    item_id: str | None = None
    cycle: str
    pass_number: str | None = Field(default=None, alias="pass")
    direction: str | None = None
    datetime: str | None = None
    variable: str | None = None


class PredictRequest(BaseModel):
    input: ModelInput
    parameters: ModelParameters = Field(default_factory=ModelParameters)
    source: SourceMetadata


def resolve_granule(request: PredictRequest) -> Path:
    """
    Resolve the MMGIS/STAC file reference to a NetCDF mounted in this service.

    Preferred deployment layout:
        /data/cycle_050/<source_file>

    The root-level fallback is useful for local test-data directories.
    """
    cycle = str(request.source.cycle).zfill(3)
    source_file = Path(request.input.source_file).name

    candidates = [
        DATA_ROOT / f"cycle_{cycle}" / source_file,
        DATA_ROOT / source_file,
    ]

    for granule in candidates:
        if granule.is_file():
            return granule

    raise HTTPException(
        status_code=404,
        detail=(
            f"Granule not found for cycle {cycle}: {source_file}. "
            f"Checked: {', '.join(str(path) for path in candidates)}"
        ),
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "swot-internal-wave-yolo",
        "model": MODEL_PATH.name,
        "filter_type": IW_YOLO_FILTER_TYPE,
    }


@app.get("/metadata")
def metadata():
    if not UI_CONFIG_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Model UI metadata not found: {UI_CONFIG_PATH}",
        )

    return json.loads(UI_CONFIG_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(request: PredictRequest):
    started = perf_counter()
    granule = resolve_granule(request)

    source_metadata = {
        "collection": request.source.collection or request.input.collection,
        "item_id": request.source.item_id or request.input.item_id or granule.stem,
        "cycle": str(request.source.cycle).zfill(3),
        "pass": request.source.pass_number,
        "direction": request.source.direction,
        "datetime": request.source.datetime,
        "variable": request.source.variable,
    }

    try:
        with TemporaryDirectory(prefix="iw_yolo_") as temp_dir:
            output_dir = Path(temp_dir)

            record = {
                "item_id": source_metadata["item_id"],
                "collection": source_metadata["collection"],
                "cycle": source_metadata["cycle"],
                "pass": source_metadata["pass"],
                "direction": source_metadata["direction"],
                "datetime": source_metadata["datetime"],
                "netcdf_path": str(granule),
            }

            output_record = run_iw_yolo_on_one_record(
                detector=detector,
                record=record,
                output_dir=output_dir,
                confidence_threshold=request.parameters.confidenceThreshold,
            )

            output_record.update({
                "collection": source_metadata["collection"],
                "direction": source_metadata["direction"],
                "datetime": source_metadata["datetime"],
            })

            features = mask_netcdf_to_features(output_record)
            processing_seconds = round(perf_counter() - started, 3)

            for feature in features:
                properties = feature.setdefault("properties", {})
                properties.pop("mask_netcdf", None)
                properties.update({
                    "source_collection": source_metadata["collection"],
                    "source_item_id": source_metadata["item_id"],
                    "cycle": source_metadata["cycle"],
                    "pass": source_metadata["pass"],
                    "direction": source_metadata["direction"],
                    "datetime": source_metadata["datetime"],
                    "processing_seconds": processing_seconds,
                })

            geojson = {
                "type": "FeatureCollection",
                "features": features,
            }

        return {
            "status": "ok",
            "service": "swot-internal-wave-yolo",
            "model_id": "iw_yolo",
            "model": MODEL_PATH.name,
            "filter_type": IW_YOLO_FILTER_TYPE,
            "confidence_threshold": request.parameters.confidenceThreshold,
            "source": source_metadata,
            "detection_count": len(features),
            "processing_seconds": processing_seconds,
            "geojson": geojson,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error