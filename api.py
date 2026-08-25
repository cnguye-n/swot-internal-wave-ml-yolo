from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from swot_internal_wave_detector import SWOTInternalWaveDetector
from pipeline.step_02_run_johnny_iw import (
    JOHNNY_FILTER_TYPE,
    run_johnny_on_one_record,
)
from pipeline.step_03_yolo_to_geojson import mask_netcdf_to_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "last.pt"


app = FastAPI(
    title="SWOT Internal Wave YOLO API",
    version="1.0.0",
)


# Load Johnny's model once when the API starts.
detector = SWOTInternalWaveDetector(
    lat_min=3.0,
    lat_max=15.5,
    model_path=MODEL_PATH,
)


class SourceMetadata(BaseModel):
    collection: str | None = None
    item_id: str | None = None
    cycle: str | None = None
    pass_number: str | None = Field(default=None, alias="pass")
    direction: str | None = None
    datetime: str | None = None


class PredictRequest(BaseModel):
    granule: str
    confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    source: SourceMetadata | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "swot-internal-wave-yolo",
        "model": MODEL_PATH.name,
        "filter_type": JOHNNY_FILTER_TYPE,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    started = perf_counter()
    granule = Path(request.granule).expanduser().resolve()

    if not granule.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Granule not found: {granule}",
        )

    source = request.source

    source_metadata = {
        "collection": source.collection if source else None,
        "item_id": source.item_id if source else granule.stem,
        "cycle": source.cycle if source else None,
        "pass": source.pass_number if source else None,
        "direction": source.direction if source else None,
        "datetime": source.datetime if source else None,
    }

    try:
        with TemporaryDirectory(prefix="johnny_iw_") as temp_dir:
            output_dir = Path(temp_dir)

            # Step 2 input.
            record = {
                "item_id": source_metadata["item_id"],
                "collection": source_metadata["collection"],
                "cycle": source_metadata["cycle"],
                "pass": source_metadata["pass"],
                "direction": source_metadata["direction"],
                "datetime": source_metadata["datetime"],
                "netcdf_path": str(granule),
            }

            # STEP 2:
            # Run Johnny's model and create the grid-aligned mask NetCDF.
            output_record = run_johnny_on_one_record(
                detector=detector,
                record=record,
                output_dir=output_dir,
                confidence_threshold=request.confidence,
            )

            # Preserve source metadata for the GeoJSON.
            output_record.update({
                "collection": source_metadata["collection"],
                "direction": source_metadata["direction"],
                "datetime": source_metadata["datetime"],
            })

            # STEP 3:
            # Convert model masks into geographic GeoJSON polygons.
            features = mask_netcdf_to_features(output_record)

            # The mask NetCDF is temporary, so don't return its path.
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
                })

            geojson = {
                "type": "FeatureCollection",
                "features": features,
            }

        return {
            "status": "ok",
            "service": "swot-internal-wave-yolo",
            "model_id": "johnny_iw",
            "model": MODEL_PATH.name,
            "filter_type": JOHNNY_FILTER_TYPE,
            "confidence_threshold": request.confidence,
            "source": source_metadata,
            "detection_count": len(features),
            "processing_seconds": round(perf_counter() - started, 3),
            "geojson": geojson,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error