from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from swot_internal_wave_detector import SWOTInternalWaveDetector


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "last.pt"


app = FastAPI(
    title="SWOT Internal Wave YOLO API",
    version="1.0.0",
)


# Load the model once when the service starts.
detector = SWOTInternalWaveDetector(
    lat_min=3.0,
    lat_max=15.5,
    model_path=MODEL_PATH,
)


class PredictRequest(BaseModel):
    granule: str

    confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
    )

    filter_type: str = "normal"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "swot-internal-wave-yolo",
        "model": MODEL_PATH.name,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    granule = Path(request.granule)

    if not granule.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Granule not found: {granule}",
        )

    allowed_filters = {
        "normal",
        "rolling",
        "gaussian",
        "stepped",
    }

    if request.filter_type not in allowed_filters:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid filter_type: {request.filter_type}. "
                f"Choose from {sorted(allowed_filters)}"
            ),
        )

    started = perf_counter()

    dataset = None

    try:
        (
            dataset,
            yolo_text,
            _boxed_image,
            result,
        ) = detector.detect_yolo_on_l3_file(
            l3_file=granule,
            filter_type=request.filter_type,
            conf=request.confidence,
            save_filtered=False,
        )

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = [
                    float(value)
                    for value in box.xyxy[0].tolist()
                ]

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names.get(
                            class_id,
                            str(class_id),
                        ),
                        "confidence": confidence,
                        "bbox_xyxy": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                    }
                )

        return {
            "status": "ok",
            "model": MODEL_PATH.name,
            "granule": granule.name,
            "filter_type": request.filter_type,
            "confidence_threshold": request.confidence,
            "detection_count": len(detections),
            "processing_seconds": round(
                perf_counter() - started,
                3,
            ),
            "detections": detections,
            "summary": yolo_text,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    finally:
        if dataset is not None:
            dataset.close()